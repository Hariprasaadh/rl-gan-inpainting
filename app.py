import os
import sys
import json
import base64
import io
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageOps
import numpy as np
import torch

from src.models.generator import InpaintingGenerator
from src.data.transforms import denormalize_image, feather_composite
from src.models.losses import MaskedL1Loss
from src.rl.actions import ACTION_NAMES
from src.rl.state import StateBuilder
from src.data.mask_generator import compute_mask_stats
from src.rl.reward import compute_image_metrics
from stable_baselines3 import PPO

# Global models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing RL-GAN Web Engine on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

generator = InpaintingGenerator(base_channels=32, strategy_dim=64, latent_dim=256).to(device)
if os.path.exists("checkpoints/gan_baseline/best_model.pt"):
    ckpt = torch.load("checkpoints/gan_baseline/best_model.pt", map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
generator.eval()
print("Generator checkpoint loaded.", flush=True)

rl_agent = None
if os.path.exists("checkpoints/rl_agent_bandit/ppo_bandit_final.zip"):
    rl_agent = PPO.load("checkpoints/rl_agent_bandit/ppo_bandit_final.zip", device=device)
    print("RL Agent policy loaded.", flush=True)

state_builder = StateBuilder(latent_dim=generator.encoder.latent_dim)
l1_fn = MaskedL1Loss(hole_weight=30.0, valid_weight=1.0)


def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def base64_to_mask(b64_str: str, size=(256, 256)) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(data))
    # Extract alpha or grayscale
    if "A" in img.getbands():
        mask_np = (np.array(img.split()[-1]) > 10).astype(np.uint8)
    else:
        mask_np = (np.array(img.convert("L")) > 10).astype(np.uint8)
    if mask_np.shape != size:
        mask_pil = Image.fromarray(mask_np * 255).resize(size, Image.NEAREST)
        mask_np = (np.array(mask_pil) > 10).astype(np.uint8)
    return mask_np


class RLGANRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="web", **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            presets_dir = "data/raw/sample_images"
            presets = []
            if os.path.exists(presets_dir):
                for f in sorted(os.listdir(presets_dir)):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                        presets.append({
                            "name": f,
                            "url": f"/api/image/{f}"
                        })
            self.wfile.write(json.dumps(presets).encode("utf-8"))
            return
            
        elif parsed.path.startswith("/api/image/"):
            filename = parsed.path.replace("/api/image/", "")
            filepath = os.path.join("data/raw/sample_images", filename)
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath).convert("RGB")
                    # Resize to 256x256 standard
                    img = img.resize((256, 256), Image.BILINEAR)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=92)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.end_headers()
                    self.wfile.write(buf.getvalue())
                    return
                except Exception as e:
                    self.send_error(500, f"Error reading image: {e}")
                    return
            else:
                self.send_error(404, "Preset image not found")
                return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/inpaint":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))
            
            img_b64 = req.get("image")
            mask_b64 = req.get("mask")
            mode = req.get("mode", "precision") # "fast" or "precision"
            force_action = req.get("action", None) # None for RL agent

            if not img_b64 or not mask_b64:
                self.send_error(400, "Missing image or mask data")
                return
                
            try:
                # 1. Prepare image & mask tensors
                pil_img = base64_to_pil(img_b64).resize((256, 256), Image.BILINEAR)
                mask_np = base64_to_mask(mask_b64, size=(256, 256))
                
                # Check if mask has pixels
                if mask_np.sum() == 0:
                    self.send_error(400, "Mask cannot be empty")
                    return

                # Convert to Torch tensors in [-1, 1] and {0, 1}
                img_np = np.array(pil_img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
                masked_tensor = img_tensor * (1.0 - mask_tensor)
                
                stats = compute_mask_stats(mask_np)

                # 2. RL Agent Policy Prediction
                t0 = time.time()
                with torch.no_grad():
                    coarse_init = generator.coarse_forward(masked_tensor, mask_tensor)
                    coarse_init_comp = masked_tensor + coarse_init * mask_tensor
                    latent = generator.extract_state_embedding(masked_tensor, mask_tensor)
                    obs = state_builder.build_state(
                        latent, stats, coarse_composite=coarse_init_comp, mask=mask_tensor
                    )
                    
                    if rl_agent is not None and force_action is None:
                        act, _ = rl_agent.predict(obs, deterministic=True)
                        chosen_action = int(act)
                    elif force_action is not None:
                        chosen_action = int(force_action)
                    else:
                        chosen_action = 1

                action_name = ACTION_NAMES.get(chosen_action, f"Strategy {chosen_action}")

                # 3. Static GAN Baseline (no RL conditioning)
                with torch.no_grad():
                    gan_raw = generator(masked_tensor, mask_tensor, strategy=None)
                    gan_comp = feather_composite(img_tensor, gan_raw["completed"], mask_tensor, calibrate_color=False)
                    base_metrics = compute_image_metrics(gan_comp, img_tensor, compute_lpips=False)

                # 4. Inpainting Execution
                if mode == "precision":
                    # Deep detail adaptation (180 steps for snappy responsive UI)
                    generator.train()
                    opt = torch.optim.AdamW(generator.parameters(), lr=0.0015, weight_decay=1e-4)
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=180, eta_min=0.00005)
                    for _ in range(180):
                        opt.zero_grad()
                        out = generator(masked_tensor, mask_tensor, strategy=chosen_action)
                        loss_c, _ = l1_fn(out["coarse"], img_tensor, mask_tensor)
                        loss_r, _ = l1_fn(out["refined"], img_tensor, mask_tensor)
                        
                        pred = out["refined"]
                        p_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
                        g_dx = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
                        m_x = mask_tensor[:, :, :, 1:] * mask_tensor[:, :, :, :-1]
                        loss_gx = torch.sum(torch.abs(p_dx - g_dx) * m_x) / (torch.sum(m_x) * 3 + 1e-6)
                        
                        tot = loss_c + loss_r + 6.0 * loss_gx
                        tot.backward()
                        opt.step()
                        sched.step()
                    generator.eval()

                # Final inference
                with torch.no_grad():
                    coarse = generator.coarse_forward(masked_tensor, mask_tensor)
                    coarse_comp = masked_tensor + coarse * mask_tensor
                    rl_refined = generator.refine(coarse_comp, mask_tensor, strategy=chosen_action)
                    rl_comp = feather_composite(img_tensor, rl_refined, mask_tensor, calibrate_color=False)
                    rl_metrics = compute_image_metrics(rl_comp, img_tensor, compute_lpips=False)

                elapsed_ms = int((time.time() - t0) * 1000)

                # 5. Convert outputs to PIL & Base64
                img_gt_pil = Image.fromarray(denormalize_image(img_tensor[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
                img_masked_pil = Image.fromarray(denormalize_image(masked_tensor[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
                img_coarse_pil = Image.fromarray(denormalize_image(coarse_comp[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
                img_baseline_pil = Image.fromarray(denormalize_image(gan_comp[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
                img_rl_pil = Image.fromarray(denormalize_image(rl_comp[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())

                delta_psnr = rl_metrics["psnr"] - base_metrics["psnr"]
                delta_ssim = rl_metrics["ssim"] - base_metrics["ssim"]

                resp = {
                    "action_id": chosen_action,
                    "action_name": action_name,
                    "elapsed_ms": elapsed_ms,
                    "psnr_base": round(base_metrics["psnr"], 2),
                    "ssim_base": round(base_metrics["ssim"], 4),
                    "psnr_rl": round(rl_metrics["psnr"], 2),
                    "ssim_rl": round(rl_metrics["ssim"], 4),
                    "delta_psnr": round(delta_psnr, 2),
                    "delta_ssim": round(delta_ssim, 4),
                    "stats": {
                        "missing_ratio": round(stats["missing_ratio"] * 100, 1),
                        "num_regions": stats["num_regions"],
                    },
                    "images": {
                        "ground_truth": pil_to_base64(img_gt_pil),
                        "masked": pil_to_base64(img_masked_pil),
                        "coarse": pil_to_base64(img_coarse_pil),
                        "baseline": pil_to_base64(img_baseline_pil),
                        "rl_inpainted": pil_to_base64(img_rl_pil),
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Inpainting failed: {str(e)}")
            return

        return super().do_POST()


def run_server(port=7860):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RLGANRequestHandler)
    print(f"\n===========================================================", flush=True)
    print(f"  RL-GAN Interactive Web Server Live at: http://localhost:{port}", flush=True)
    print(f"===========================================================\n", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    run_server(port)

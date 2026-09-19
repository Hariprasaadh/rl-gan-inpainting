import os
import gdown
import torch

# nipponjo/deepfillv2-pytorch pretrained weights on Google Drive:
# States PT (PyTorch native): 1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa
# States TF (TF converted):   1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt
GDRIVE_FILE_ID = "1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa"
DEST_PATH = os.path.join("checkpoints", "pretrained", "deepfillv2_places2.pth")


def download_pretrained_gan():
    os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)
    print(f"Downloading DeepFillv2 checkpoint to: {DEST_PATH}...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    output = gdown.download(url, DEST_PATH, quiet=False)

    if output and os.path.exists(DEST_PATH):
        try:
            ckpt = torch.load(DEST_PATH, map_location="cpu", weights_only=False)
            print(f"Validation SUCCESS! Loaded checkpoint with {len(ckpt) if isinstance(ckpt, dict) else 'weights'}")
        except Exception as e:
            print(f"Validation FAILED: {e}")
            print(f"File size: {os.path.getsize(DEST_PATH)} bytes")
    else:
        print("Download failed.")


if __name__ == "__main__":
    download_pretrained_gan()


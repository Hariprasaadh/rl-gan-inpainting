// RL-GAN Studio Frontend Application Logic

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const sourceImage = document.getElementById("sourceImage");
  const maskCanvas = document.getElementById("maskCanvas");
  const ctx = maskCanvas.getContext("2d");
  const canvasContainer = document.getElementById("canvasContainer");
  const canvasCursor = document.getElementById("canvasCursor");
  const canvasBadge = document.getElementById("canvasBadge");
  const brushSizeInput = document.getElementById("brushSize");
  const brushSizeVal = document.getElementById("brushSizeVal");
  const btnClearMask = document.getElementById("btnClearMask");
  const btnRandomMask = document.getElementById("btnRandomMask");
  const btnAutoDetectTears = document.getElementById("btnAutoDetectTears");
  const btnModeSynthetic = document.getElementById("btnModeSynthetic");
  const btnModeRestoration = document.getElementById("btnModeRestoration");
  const btnInpaint = document.getElementById("btnInpaint");
  const fileUpload = document.getElementById("fileUpload");
  const presetCards = document.getElementById("presetCards");

  // Dynamic Labels
  const sliderLabelLeft = document.getElementById("sliderLabelLeft");
  const sliderLabelRight = document.getElementById("sliderLabelRight");
  const pipeTag1 = document.getElementById("pipeTag1");
  const pipeTag2 = document.getElementById("pipeTag2");
  const statTitle1 = document.getElementById("statTitle1");
  const statTitle2 = document.getElementById("statTitle2");
  const gainPill = document.getElementById("gainPill");

  // Results & Split Slider Elements
  const splitSliderBox = document.getElementById("splitSliderBox");
  const splitResultImg = document.getElementById("splitResultImg");
  const splitDamagedImg = document.getElementById("splitDamagedImg");
  const splitDamagedWrapper = document.getElementById("splitDamagedWrapper");
  const splitDivider = document.getElementById("splitDivider");
  const btnViewSplit = document.getElementById("btnViewSplit");
  const btnViewPipeline = document.getElementById("btnViewPipeline");
  const splitViewContainer = document.getElementById("splitViewContainer");
  const pipelineViewContainer = document.getElementById("pipelineViewContainer");

  // Pipeline Images
  const pipeGt = document.getElementById("pipeGt");
  const pipeMasked = document.getElementById("pipeMasked");
  const pipeCoarse = document.getElementById("pipeCoarse");
  const pipeBaseline = document.getElementById("pipeBaseline");
  const pipeRl = document.getElementById("pipeRl");

  // Telemetry Elements
  const decisionAction = document.getElementById("decisionAction");
  const decisionDetail = document.getElementById("decisionDetail");
  const deltaPsnrVal = document.getElementById("deltaPsnrVal");
  const psnrRlVal = document.getElementById("psnrRlVal");
  const psnrBaseSub = document.getElementById("psnrBaseSub");
  const ssimRlVal = document.getElementById("ssimRlVal");
  const ssimBaseSub = document.getElementById("ssimBaseSub");
  const coverageVal = document.getElementById("coverageVal");
  const latencyVal = document.getElementById("latencyVal");

  // State
  let workflowMode = "synthetic"; // "synthetic" | "restoration"
  let isDrawing = false;
  let brushSize = parseInt(brushSizeInput.value, 10);
  let isDraggingSlider = false;

  // Initialize Canvas
  function clearMask() {
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  }

  // Workflow Mode Switcher
  function setWorkflowMode(mode, autoDetect = false) {
    workflowMode = mode;
    if (mode === "restoration") {
      btnModeRestoration.classList.add("active");
      btnModeSynthetic.classList.remove("active");
      if (canvasBadge) {
        canvasBadge.textContent = "Restoration Studio";
        canvasBadge.style.borderColor = "var(--accent-cyan)";
        canvasBadge.style.color = "var(--accent-cyan)";
      }
      if (btnRandomMask) btnRandomMask.style.display = "none";
      if (sliderLabelLeft) sliderLabelLeft.textContent = "Original Damaged Photo";
      if (sliderLabelRight) sliderLabelRight.textContent = "RL-GAN Restored";
      if (pipeTag1) pipeTag1.textContent = "1. Damaged Photo";
      if (pipeTag2) pipeTag2.textContent = "2. Masked Tears";
      if (statTitle1) statTitle1.textContent = "Restoration Mode";
      if (statTitle2) statTitle2.textContent = "PPO Strategy";
      if (autoDetect) {
        autoDetectTears();
      }
    } else {
      btnModeSynthetic.classList.add("active");
      btnModeRestoration.classList.remove("active");
      if (canvasBadge) {
        canvasBadge.textContent = "Canvas";
        canvasBadge.style.borderColor = "var(--border-color)";
        canvasBadge.style.color = "var(--text-muted)";
      }
      if (btnRandomMask) btnRandomMask.style.display = "inline-flex";
      if (sliderLabelLeft) sliderLabelLeft.textContent = "Damaged Input";
      if (sliderLabelRight) sliderLabelRight.textContent = "RL-GAN Reconstructed";
      if (pipeTag1) pipeTag1.textContent = "1. Ground Truth";
      if (pipeTag2) pipeTag2.textContent = "2. Masked Input";
      if (statTitle1) statTitle1.textContent = "RL-GAN PSNR";
      if (statTitle2) statTitle2.textContent = "Structural Similarity";
    }
  }

  btnModeSynthetic.addEventListener("click", () => setWorkflowMode("synthetic"));
  btnModeRestoration.addEventListener("click", () => setWorkflowMode("restoration", true));

  // Auto-Detect Tears & Scratches
  async function autoDetectTears() {
    if (btnAutoDetectTears) {
      btnAutoDetectTears.classList.add("loading");
      btnAutoDetectTears.textContent = "⏳ Detecting...";
    }
    try {
      const imgB64 = getSourceImageBase64();
      const res = await fetch("/api/detect-defects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgB64, sensitivity: "medium" })
      });
      if (!res.ok) {
        console.error("Failed to detect defects");
        return;
      }
      const data = await res.json();
      const maskImg = new Image();
      maskImg.onload = () => {
        clearMask();
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = 256;
        tempCanvas.height = 256;
        const tempCtx = tempCanvas.getContext("2d");
        tempCtx.drawImage(maskImg, 0, 0, 256, 256);
        const imgData = tempCtx.getImageData(0, 0, 256, 256);

        const canvasData = ctx.createImageData(256, 256);
        for (let i = 0; i < imgData.data.length; i += 4) {
          if (imgData.data[i] > 120) {
            canvasData.data[i] = 239;     // R
            canvasData.data[i + 1] = 68;  // G
            canvasData.data[i + 2] = 68;  // B
            canvasData.data[i + 3] = 190; // Alpha
          }
        }
        ctx.putImageData(canvasData, 0, 0);

        decisionAction.textContent = `Tears Detected (${data.coverage}%)`;
        decisionDetail.textContent = `Identified ${data.num_defects} defect regions. You can click 'Run RL-GAN' or use the brush to refine.`;
      };
      maskImg.src = data.mask;
    } catch (err) {
      console.error("Defect detection error:", err);
    } finally {
      if (btnAutoDetectTears) {
        btnAutoDetectTears.classList.remove("loading");
        btnAutoDetectTears.textContent = "⚡ Auto-Detect Tears";
      }
    }
  }

  if (btnAutoDetectTears) {
    btnAutoDetectTears.addEventListener("click", autoDetectTears);
  }

  // Brush Size
  brushSizeInput.addEventListener("input", (e) => {
    brushSize = parseInt(e.target.value, 10);
    brushSizeVal.textContent = `${brushSize}px`;
    updateCursorSize();
  });

  function updateCursorSize() {
    const scale = canvasContainer.clientWidth / maskCanvas.width;
    const visualSize = brushSize * scale;
    canvasCursor.style.width = `${visualSize}px`;
    canvasCursor.style.height = `${visualSize}px`;
  }

  // Canvas Mouse & Touch Drawing
  function getCanvasCoords(e) {
    const rect = canvasContainer.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
      screenX: clientX - rect.left,
      screenY: clientY - rect.top
    };
  }

  function startDraw(e) {
    isDrawing = true;
    const coords = getCanvasCoords(e);
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(239, 68, 68, 0.75)";
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
  }

  function drawMove(e) {
    const coords = getCanvasCoords(e);

    // Update cursor circle
    canvasCursor.style.display = "block";
    canvasCursor.style.left = `${coords.screenX}px`;
    canvasCursor.style.top = `${coords.screenY}px`;

    if (!isDrawing) return;
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
  }

  function stopDraw() {
    if (isDrawing) {
      ctx.closePath();
      isDrawing = false;
    }
  }

  canvasContainer.addEventListener("mousedown", startDraw);
  canvasContainer.addEventListener("mousemove", drawMove);
  window.addEventListener("mouseup", stopDraw);

  canvasContainer.addEventListener("touchstart", (e) => { e.preventDefault(); startDraw(e); });
  canvasContainer.addEventListener("touchmove", (e) => { e.preventDefault(); drawMove(e); });
  canvasContainer.addEventListener("touchend", stopDraw);

  canvasContainer.addEventListener("mouseenter", () => {
    updateCursorSize();
    canvasCursor.style.display = "block";
  });
  canvasContainer.addEventListener("mouseleave", () => {
    canvasCursor.style.display = "none";
  });

  btnClearMask.addEventListener("click", clearMask);

  // Generate Organic Random Mask
  btnRandomMask.addEventListener("click", () => {
    clearMask();
    ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const pad = 40;
    const w = maskCanvas.width;
    const h = maskCanvas.height;

    const strokes = 2 + Math.floor(Math.random() * 2);
    for (let s = 0; s < strokes; s++) {
      ctx.lineWidth = 14 + Math.random() * 10;
      let x = pad + Math.random() * (w - pad * 2);
      let y = pad + Math.random() * (h - pad * 2);
      ctx.beginPath();
      ctx.moveTo(x, y);
      const points = 2 + Math.floor(Math.random() * 3);
      for (let p = 0; p < points; p++) {
        x = pad + Math.random() * (w - pad * 2);
        y = pad + Math.random() * (h - pad * 2);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  });

  // Fetch & Render Presets
  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      const presets = await res.json();
      presetCards.innerHTML = "";

      presets.forEach((p, idx) => {
        const isDamaged = p.name.toLowerCase().includes("damaged") || p.name.toLowerCase().includes("torn");
        const card = document.createElement("div");
        card.className = `preset-card ${idx === 0 ? "active" : ""} ${isDamaged ? "preset-damaged" : ""}`;
        card.innerHTML = `<img src="${p.url}" alt="${p.name}">${isDamaged ? '<span class="preset-badge">TORN</span>' : ''}`;
        card.addEventListener("click", () => {
          document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
          card.classList.add("active");
          setImageSource(p.url, isDamaged);
        });
        presetCards.appendChild(card);
      });

      if (presets.length > 0) {
        const firstIsDamaged = presets[0].name.toLowerCase().includes("damaged") || presets[0].name.toLowerCase().includes("torn");
        setImageSource(presets[0].url, firstIsDamaged);
      }
    } catch (err) {
      console.error("Failed to load presets", err);
    }
  }

  function setImageSource(url, isDamaged = false) {
    sourceImage.src = url;
    sourceImage.onload = () => {
      clearMask();
      if (isDamaged) {
        setWorkflowMode("restoration");
        autoDetectTears();
      } else {
        if (workflowMode === "synthetic") {
          btnRandomMask.click(); // Auto-add a defect for initial visual demo
        }
      }
    };
  }

  // Custom File Upload
  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const isDamaged = file.name.toLowerCase().includes("damaged") || file.name.toLowerCase().includes("torn");
    const reader = new FileReader();
    reader.onload = (event) => {
      document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
      setImageSource(event.target.result, isDamaged || workflowMode === "restoration");
    };
    reader.readAsDataURL(file);
  });

  // Tab View Toggling
  btnViewSplit.addEventListener("click", () => {
    btnViewSplit.classList.add("active");
    btnViewPipeline.classList.remove("active");
    splitViewContainer.style.display = "flex";
    pipelineViewContainer.style.display = "none";
  });

  btnViewPipeline.addEventListener("click", () => {
    btnViewPipeline.classList.add("active");
    btnViewSplit.classList.remove("active");
    splitViewContainer.style.display = "none";
    pipelineViewContainer.style.display = "flex";
  });

  // Interactive Before/After Split Slider
  function updateSliderPosition(clientX) {
    const rect = splitSliderBox.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const percent = (x / rect.width) * 100;

    splitDivider.style.left = `${percent}%`;
    splitDamagedWrapper.style.width = `${percent}%`;
  }

  splitSliderBox.addEventListener("mousedown", (e) => {
    isDraggingSlider = true;
    updateSliderPosition(e.clientX);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDraggingSlider) return;
    updateSliderPosition(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDraggingSlider = false;
  });

  splitSliderBox.addEventListener("touchstart", (e) => {
    isDraggingSlider = true;
    updateSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDraggingSlider) return;
    updateSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchend", () => {
    isDraggingSlider = false;
  });

  // Get Clean Base64 of Source Image
  function getSourceImageBase64() {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 256;
    tempCanvas.height = 256;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(sourceImage, 0, 0, 256, 256);
    return tempCanvas.toDataURL("image/png");
  }

  // Get Mask as Binary White-on-Black Base64
  function getMaskBase64() {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 256;
    tempCanvas.height = 256;
    const tempCtx = tempCanvas.getContext("2d");

    tempCtx.fillStyle = "#000000";
    tempCtx.fillRect(0, 0, 256, 256);

    const maskData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const binaryData = tempCtx.createImageData(256, 256);

    for (let i = 0; i < maskData.data.length; i += 4) {
      const alpha = maskData.data[i + 3];
      const val = alpha > 10 ? 255 : 0;
      binaryData.data[i] = val;
      binaryData.data[i + 1] = val;
      binaryData.data[i + 2] = val;
      binaryData.data[i + 3] = val;
    }
    tempCtx.putImageData(binaryData, 0, 0);
    return tempCanvas.toDataURL("image/png");
  }

  // Trigger Inpainting API
  async function runInpainting() {
    btnInpaint.classList.add("loading");
    btnInpaint.disabled = true;

    try {
      const imgB64 = getSourceImageBase64();
      const maskB64 = getMaskBase64();
      const mode = document.querySelector('input[name="inferMode"]:checked').value;

      const res = await fetch("/api/inpaint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imgB64,
          mask: maskB64,
          mode: mode,
          workflow_mode: workflowMode,
          is_restoration: workflowMode === "restoration"
        })
      });

      if (!res.ok) {
        const errData = await res.text();
        alert(`Inpainting error: ${errData}`);
        return;
      }

      const data = await res.json();

      if (data.is_restoration) {
        // Authentic Photo Restoration Mode: Show Original Damaged Photo vs Restored Result
        splitResultImg.src = data.images.rl_inpainted;
        splitDamagedImg.src = data.images.original_damaged || data.images.ground_truth;

        // 5-Stage Pipeline Cards
        pipeGt.src = data.images.original_damaged || data.images.ground_truth;
        pipeMasked.src = data.images.masked;
        pipeCoarse.src = data.images.coarse;
        pipeBaseline.src = data.images.baseline;
        pipeRl.src = data.images.rl_inpainted;

        // Decision Telemetry
        decisionAction.textContent = data.action_name;
        decisionDetail.textContent = `PPO bandit policy evaluated 261 features of the torn image and selected the ${data.action_name} adapter.`;

        deltaPsnrVal.textContent = "Restored";
        if (gainPill) {
          const lbl = gainPill.querySelector(".gain-label");
          if (lbl) lbl.textContent = "Status";
        }
        psnrRlVal.textContent = "Restored Photo";
        psnrBaseSub.textContent = "Authentic Scan (No GT)";

        ssimRlVal.textContent = data.action_name;
        ssimBaseSub.textContent = `Action ${data.action_id} Selected`;

        coverageVal.textContent = `${data.stats.missing_ratio}%`;
        latencyVal.textContent = `${data.elapsed_ms} ms`;
      } else {
        // Standard Synthetic Benchmark Mode
        splitResultImg.src = data.images.rl_inpainted;
        splitDamagedImg.src = data.images.masked;

        pipeGt.src = data.images.ground_truth;
        pipeMasked.src = data.images.masked;
        pipeCoarse.src = data.images.coarse;
        pipeBaseline.src = data.images.baseline;
        pipeRl.src = data.images.rl_inpainted;

        decisionAction.textContent = data.action_name;
        decisionDetail.textContent = `The PPO policy evaluated 261 visual features and dynamically modulated FiLM conditioning for ${data.action_name.toLowerCase()}.`;

        deltaPsnrVal.textContent = `${data.delta_psnr >= 0 ? "+" : ""}${data.delta_psnr.toFixed(1)} dB`;
        if (gainPill) {
          const lbl = gainPill.querySelector(".gain-label");
          if (lbl) lbl.textContent = "PSNR Gain";
        }
        psnrRlVal.textContent = `${data.psnr_rl.toFixed(1)} dB`;
        psnrBaseSub.textContent = `Baseline: ${data.psnr_base.toFixed(1)} dB`;

        ssimRlVal.textContent = data.ssim_rl.toFixed(4);
        ssimBaseSub.textContent = `Baseline: ${data.ssim_base.toFixed(4)}`;

        coverageVal.textContent = `${data.stats.missing_ratio}%`;
        latencyVal.textContent = `${data.elapsed_ms} ms`;
      }

      // Match damaged layer width to container
      splitDamagedWrapper.querySelector(".damaged-layer").style.width = `${splitSliderBox.clientWidth}px`;

    } catch (err) {
      console.error("Failed to run inpainting", err);
      alert("Inpainting request failed. Check server console.");
    } finally {
      btnInpaint.classList.remove("loading");
      btnInpaint.disabled = false;
    }
  }

  btnInpaint.addEventListener("click", runInpainting);

  // Resize damaged layer with container on window resize
  window.addEventListener("resize", () => {
    const layer = splitDamagedWrapper.querySelector(".damaged-layer");
    if (layer) {
      layer.style.width = `${splitSliderBox.clientWidth}px`;
    }
  });

  // Start
  loadPresets();
});

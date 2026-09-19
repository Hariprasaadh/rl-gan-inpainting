// RL-GAN Studio Frontend Application Logic

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const sourceImage = document.getElementById("sourceImage");
  const maskCanvas = document.getElementById("maskCanvas");
  const ctx = maskCanvas.getContext("2d");
  const canvasContainer = document.getElementById("canvasContainer");
  const canvasCursor = document.getElementById("canvasCursor");
  const brushSizeInput = document.getElementById("brushSize");
  const brushSizeVal = document.getElementById("brushSizeVal");
  const btnClearMask = document.getElementById("btnClearMask");
  const btnRandomMask = document.getElementById("btnRandomMask");
  const btnInpaint = document.getElementById("btnInpaint");
  const fileUpload = document.getElementById("fileUpload");
  const presetCards = document.getElementById("presetCards");

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
  let isDrawing = false;
  let brushSize = parseInt(brushSizeInput.value, 10);
  let isDraggingSlider = false;

  // Initialize Canvas
  function clearMask() {
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
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
        const card = document.createElement("div");
        card.className = `preset-card ${idx === 0 ? "active" : ""}`;
        card.innerHTML = `<img src="${p.url}" alt="${p.name}">`;
        card.addEventListener("click", () => {
          document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
          card.classList.add("active");
          setImageSource(p.url);
        });
        presetCards.appendChild(card);
      });

      if (presets.length > 0) {
        setImageSource(presets[0].url);
      }
    } catch (err) {
      console.error("Failed to load presets", err);
    }
  }

  function setImageSource(url) {
    sourceImage.src = url;
    sourceImage.onload = () => {
      clearMask();
      btnRandomMask.click(); // Auto-add a defect for initial visual demo
    };
  }

  // Custom File Upload
  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
      setImageSource(event.target.result);
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
          mode: mode
        })
      });

      if (!res.ok) {
        const errData = await res.text();
        alert(`Inpainting error: ${errData}`);
        return;
      }

      const data = await res.json();

      // Update Slider Images
      splitResultImg.src = data.images.rl_inpainted;
      splitDamagedImg.src = data.images.masked;

      // Match damaged layer width to container
      splitDamagedWrapper.querySelector(".damaged-layer").style.width = `${splitSliderBox.clientWidth}px`;

      // Update 5 Pipeline Images
      pipeGt.src = data.images.ground_truth;
      pipeMasked.src = data.images.masked;
      pipeCoarse.src = data.images.coarse;
      pipeBaseline.src = data.images.baseline;
      pipeRl.src = data.images.rl_inpainted;

      // Update Telemetry & Decision Card
      decisionAction.textContent = data.action_name;
      decisionDetail.textContent = `The PPO policy evaluated 261 visual features and dynamically modulated FiLM conditioning for ${data.action_name.toLowerCase()}.`;

      deltaPsnrVal.textContent = `${data.delta_psnr >= 0 ? "+" : ""}${data.delta_psnr.toFixed(1)} dB`;
      psnrRlVal.textContent = `${data.psnr_rl.toFixed(1)} dB`;
      psnrBaseSub.textContent = `Baseline: ${data.psnr_base.toFixed(1)} dB`;

      ssimRlVal.textContent = data.ssim_rl.toFixed(4);
      ssimBaseSub.textContent = `Baseline: ${data.ssim_base.toFixed(4)}`;

      coverageVal.textContent = `${data.stats.missing_ratio}%`;
      latencyVal.textContent = `${data.elapsed_ms} ms`;

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

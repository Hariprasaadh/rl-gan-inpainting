// RL-GAN Studio Application Logic

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
  const dropzone = document.getElementById("dropzone");
  const presetCards = document.getElementById("presetCards");

  // View Switching Tabs
  const tabGrid = document.getElementById("tabGrid");
  const tabSlider = document.getElementById("tabSlider");
  const gridView = document.getElementById("gridView");
  const sliderView = document.getElementById("sliderView");

  // 4-Panel Stage Images
  const imgGt = document.getElementById("imgGt");
  const imgMasked = document.getElementById("imgMasked");
  const imgBaseline = document.getElementById("imgBaseline");
  const imgRl = document.getElementById("imgRl");

  const cardCoverage = document.getElementById("cardCoverage");
  const cardBasePsnr = document.getElementById("cardBasePsnr");
  const cardRlPsnr = document.getElementById("cardRlPsnr");
  const tagActionName = document.getElementById("tagActionName");

  // Split Slider Elements
  const splitSliderBox = document.getElementById("splitSliderBox");
  const splitResultImg = document.getElementById("splitResultImg");
  const splitDamagedImg = document.getElementById("splitDamagedImg");
  const splitDamagedWrapper = document.getElementById("splitDamagedWrapper");
  const splitDivider = document.getElementById("splitDivider");

  // Telemetry Elements
  const rlActionTitle = document.getElementById("rlActionTitle");
  const rlActionDesc = document.getElementById("rlActionDesc");
  const telemetryDeltaPsnr = document.getElementById("telemetryDeltaPsnr");
  const telemetryPsnr = document.getElementById("telemetryPsnr");
  const telemetryBasePsnr = document.getElementById("telemetryBasePsnr");
  const telemetrySsim = document.getElementById("telemetrySsim");
  const telemetryBaseSsim = document.getElementById("telemetryBaseSsim");
  const telemetryCoverage = document.getElementById("telemetryCoverage");
  const telemetryLatency = document.getElementById("telemetryLatency");

  // State Variables
  let isDrawing = false;
  let brushSize = parseInt(brushSizeInput.value, 10);
  let isDraggingSlider = false;

  // Clear Mask
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
    ctx.fillStyle = "rgba(239, 68, 68, 0.85)";
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
  }

  function drawMove(e) {
    const coords = getCanvasCoords(e);

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
    ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
    ctx.fillStyle = "rgba(239, 68, 68, 0.85)";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const pad = 40;
    const w = maskCanvas.width;
    const h = maskCanvas.height;

    const strokes = 2 + Math.floor(Math.random() * 2);
    for (let s = 0; s < strokes; s++) {
      ctx.lineWidth = 14 + Math.random() * 8;
      let x = pad + Math.random() * (w - pad * 2);
      let y = pad + Math.random() * (h - pad * 2);
      ctx.beginPath();
      ctx.moveTo(x, y);
      const points = 2 + Math.floor(Math.random() * 2);
      for (let p = 0; p < points; p++) {
        x = pad + Math.random() * (w - pad * 2);
        y = pad + Math.random() * (h - pad * 2);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  });

  // Drag and Drop Upload Support
  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    }, false);
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    }, false);
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const file = dt.files[0];
    if (file && file.type.startsWith("image/")) {
      handleImageFile(file);
    }
  });

  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      handleImageFile(file);
    }
  });

  function handleImageFile(file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("active"));
      setImageSource(event.target.result);
    };
    reader.readAsDataURL(file);
  }

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
      btnRandomMask.click(); // Auto-add defect mask for instant visual gratification
      // Preload Ground Truth slot
      imgGt.src = url;
    };
  }

  // Tab View Toggling
  tabGrid.addEventListener("click", () => {
    tabGrid.classList.add("active");
    tabSlider.classList.remove("active");
    gridView.style.display = "block";
    sliderView.style.display = "none";
  });

  tabSlider.addEventListener("click", () => {
    tabSlider.classList.add("active");
    tabGrid.classList.remove("active");
    gridView.style.display = "none";
    sliderView.style.display = "block";
    updateSliderLayerWidth();
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

  function updateSliderLayerWidth() {
    const layer = splitDamagedWrapper.querySelector(".damaged-layer");
    if (layer) {
      layer.style.width = `${splitSliderBox.clientWidth}px`;
    }
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

  window.addEventListener("resize", updateSliderLayerWidth);

  // Convert Source Image to Base64 (256x256)
  function getSourceImageBase64() {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 256;
    tempCanvas.height = 256;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(sourceImage, 0, 0, 256, 256);
    return tempCanvas.toDataURL("image/png");
  }

  // Convert Drawn Mask to Binary White-on-Black Base64
  function getMaskBase64() {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 256;
    tempCanvas.height = 256;
    const tempCtx = tempCanvas.getContext("2d");

    tempCtx.fillStyle = "#000000";
    tempCtx.fillRect(0, 0, 256, 256);

    const maskData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const binaryData = tempCtx.createImageData(256, 256);

    let strokeCount = 0;
    for (let i = 0; i < maskData.data.length; i += 4) {
      const alpha = maskData.data[i + 3];
      const val = (alpha > 10 || maskData.data[i] > 100) ? 255 : 0;
      if (val === 255) strokeCount++;
      binaryData.data[i] = val;
      binaryData.data[i + 1] = val;
      binaryData.data[i + 2] = val;
      binaryData.data[i + 3] = 255;
    }
    tempCtx.putImageData(binaryData, 0, 0);
    return { dataUrl: tempCanvas.toDataURL("image/png"), strokeCount };
  }

  // Trigger Inpainting API
  async function runInpainting() {
    const maskInfo = getMaskBase64();
    if (maskInfo.strokeCount < 10) {
      alert("Please draw damage on the image with the paintbrush, or click '✨ Auto Defect' to create a mask!");
      return;
    }

    btnInpaint.classList.add("loading");
    btnInpaint.disabled = true;

    try {
      const imgB64 = getSourceImageBase64();
      const maskB64 = maskInfo.dataUrl;
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

      // Update EXACTLY the 4 Stages requested by user (NO coarse pass!)
      // 1. Ground Truth
      imgGt.src = data.images.ground_truth;
      // 2. Masked Input
      imgMasked.src = data.images.masked;
      cardCoverage.textContent = `Mask Area: ${data.stats.missing_ratio}%`;
      // 3. Baseline GAN
      imgBaseline.src = data.images.baseline;
      cardBasePsnr.textContent = `PSNR: ${data.psnr_base} dB (SSIM: ${data.ssim_base})`;
      // 4. RL-GAN
      imgRl.src = data.images.rl_inpainted;
      cardRlPsnr.textContent = `PSNR: ${data.psnr_rl} dB (Gain: +${data.delta_psnr} dB)`;
      tagActionName.textContent = data.action_name;

      // Update Interactive Slider Images
      splitResultImg.src = data.images.rl_inpainted;
      splitDamagedImg.src = data.images.masked;
      updateSliderLayerWidth();

      // Update Telemetry & Decision Banner
      rlActionTitle.textContent = data.action_name;
      rlActionDesc.textContent = `PPO controller observed 261-D state embedding and modulated GAN FiLM layers for optimal ${data.action_name.toLowerCase()}.`;

      telemetryDeltaPsnr.textContent = `${data.delta_psnr >= 0 ? "+" : ""}${data.delta_psnr.toFixed(1)} dB`;
      telemetryPsnr.textContent = `${data.psnr_rl.toFixed(1)} dB`;
      telemetryBasePsnr.textContent = `Baseline: ${data.psnr_base.toFixed(1)} dB`;

      telemetrySsim.textContent = data.ssim_rl.toFixed(4);
      telemetryBaseSsim.textContent = `Baseline: ${data.ssim_base.toFixed(4)}`;

      telemetryCoverage.textContent = `${data.stats.missing_ratio}%`;
      telemetryLatency.textContent = `${data.elapsed_ms} ms`;

    } catch (err) {
      console.error("Failed to execute inpainting", err);
      alert("Inpainting execution failed. Check console.");
    } finally {
      btnInpaint.classList.remove("loading");
      btnInpaint.disabled = false;
    }
  }

  btnInpaint.addEventListener("click", runInpainting);

  // Initialize
  loadPresets();
});

// Scribe: free in-browser English transcription using Whisper (Transformers.js).
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const $ = (id) => document.getElementById(id);
const t00 = performance.now();
function log(msg) {
  const el = $("log");
  if (el) el.textContent += `[${((performance.now() - t00) / 1000).toFixed(1)}s] ${msg}\n`;
  console.log(msg);
}
addEventListener("error", (e) => log("Error: " + (e.message || e)));
addEventListener("unhandledrejection", (e) => log("Error: " + (e.reason?.message || e.reason)));
const fileInput = $("fileInput"), dropzone = $("dropzone"), dropText = $("dropText");
const player = $("player"), goBtn = $("goBtn"), modelSelect = $("modelSelect");
const statusBox = $("status"), statusText = $("statusText"), barFill = $("barFill");
const output = $("output"), copyBtn = $("copyBtn"), saveBtn = $("saveBtn"), tsToggle = $("tsToggle");
const canvas = $("wave"), ctx = canvas.getContext("2d");

let currentFile = null;
let transcriber = null, loadedModel = null, lib = null;
let segments = [];
let audioData = null;
let wakeLock = null;

// ---------- Upload handling ----------
fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });

async function setFile(file) {
  currentFile = file;
  dropText.innerHTML = `${escapeHtml(file.name)}<br><small>${(file.size / 1048576).toFixed(1)} MB. Click to choose another file.</small>`;
  player.src = URL.createObjectURL(file);
  player.hidden = false;
  goBtn.disabled = false;
  setStatus("", false);
  audioData = null;
  try {
    audioData = await decodeAudio(file);
    drawWave(audioData);
  } catch { clearWave(); }
}

// ---------- Audio decoding (to 16 kHz mono, as Whisper expects) ----------
async function decodeAudio(file) {
  const buf = await file.arrayBuffer();
  const actx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const decoded = await actx.decodeAudioData(buf);
  let data;
  if (decoded.numberOfChannels > 1) {
    const a = decoded.getChannelData(0), b = decoded.getChannelData(1);
    data = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) data[i] = (a[i] + b[i]) / 2;
  } else data = decoded.getChannelData(0).slice();
  actx.close();
  return data;
}

// ---------- Waveform ----------
function clearWave() { ctx.clearRect(0, 0, canvas.width, canvas.height); }
function drawWave(data) {
  clearWave();
  const w = canvas.width, h = canvas.height, step = Math.floor(data.length / w) || 1;
  const g = ctx.createLinearGradient(0, 0, w, 0);
  g.addColorStop(0, "#f6a5c0"); g.addColorStop(1, "#ffd0a1");
  ctx.fillStyle = g;
  for (let x = 0; x < w; x += 3) {
    let max = 0;
    for (let i = 0; i < step * 3; i++) max = Math.max(max, Math.abs(data[x * step + i] || 0));
    const bar = Math.max(2, max * h);
    ctx.fillRect(x, (h - bar) / 2, 2, bar);
  }
}

// ---------- Model loading ----------
// Resolves with the loaded model, or rejects with a clear message if setup stalls.
async function getTranscriber(modelId, useGpu) {
  const key = modelId + (useGpu ? "|gpu" : "");
  if (transcriber && loadedModel === key) return transcriber;

  setStatus("Loading the AI library…", true);
  log("Loading library from the CDN");
  if (!lib) {
    lib = await Promise.race([
      import(TRANSFORMERS_URL),
      new Promise((_, rej) => setTimeout(() => rej(new Error("The AI library could not be loaded. Check your internet connection or turn off ad blockers for this page.")), 60000)),
    ]);
  }
  log("Library loaded");
  lib.env.allowLocalModels = false;
  // Some browsers refuse the cache when a page is opened straight from a folder. Skip it if so.
  try { await caches.open("scribe-test"); } catch { lib.env.useBrowserCache = false; log("Browser cache unavailable, model will not be remembered"); }

  const files = {};
  let setupStart = 0;
  const progress_callback = (p) => {
    if (p.status === "initiate") log("Fetching " + p.file);
    if (p.status === "progress" && p.file) {
      files[p.file] = { loaded: p.loaded || 0, total: p.total || 0 };
      const all = Object.values(files);
      const total = all.reduce((a, f) => a + f.total, 0), loaded = all.reduce((a, f) => a + f.loaded, 0);
      const pct = total ? (loaded / total) * 100 : p.progress;
      setStatus(`Downloading the AI model… ${Math.round(pct)}% (${(loaded / 1048576).toFixed(0)} of ${(total / 1048576).toFixed(0)} MB)`, false, pct);
      if (pct >= 99.5 && !setupStart) { setupStart = performance.now(); log("Download finished, starting the model"); }
    } else if (p.status === "ready") log("Model ready");
  };
  // While the model starts up, there are no events, so keep the person informed.
  const ticker = setInterval(() => {
    if (setupStart) setStatus(`Starting the model… ${Math.round((performance.now() - setupStart) / 1000)}s. This can take up to a minute the first time.`, true);
  }, 1000);

  const task = "automatic-speech-recognition";
  try {
    if (useGpu && navigator.gpu) {
      try {
        log("Trying GPU mode");
        transcriber = await lib.pipeline(task, modelId, {
          device: "webgpu", dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" }, progress_callback,
        });
        loadedModel = key; log("GPU mode ready");
        return transcriber;
      } catch (e) {
        log("GPU mode failed: " + (e.message || e));
        setStatus("GPU mode isn't available here. Using the processor instead…", true);
      }
    }
    log("Using processor mode");
    transcriber = await lib.pipeline(task, modelId, { progress_callback });
    loadedModel = key;
    return transcriber;
  } finally { clearInterval(ticker); }
}

// ---------- Transcribe ----------
const SR = 16000, PIECE = 120 * SR; // work in ~2 minute pieces

// Cut at the quietest moment near a boundary so words aren't split in half.
function quietPoint(audio, end) {
  const frame = 1600, from = Math.max(0, end - 3 * SR);
  let best = end, bestE = Infinity;
  for (let i = from; i + frame <= end; i += frame) {
    let e = 0;
    for (let j = i; j < i + frame; j++) e += audio[j] * audio[j];
    if (e < bestE) { bestE = e; best = i + frame / 2; }
  }
  return Math.floor(best);
}
function fmtEta(sec) {
  if (!isFinite(sec) || sec < 1) return "less than a minute";
  const m = Math.round(sec / 60);
  return m < 1 ? "less than a minute" : `about ${m} min`;
}

goBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  goBtn.disabled = true; copyBtn.disabled = saveBtn.disabled = true;
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
  try {
    const t = await getTranscriber(modelSelect.value, $("gpuToggle").checked);
    log("Reading audio");
    setStatus("Reading your audio…", true);
    const audio = audioData || (audioData = await decodeAudio(currentFile));
    segments = []; output.classList.remove("empty"); output.innerHTML = "";
    const total = audio.length, t0 = performance.now();
    let pos = 0;
    heroBusy = true;
    log("Transcribing " + (total / SR / 60).toFixed(1) + " minutes of audio");
    setStatus("Transcribing the first part… the percentage appears when it finishes.", true);
    while (pos < total) {
      let end = Math.min(pos + (pos === 0 ? 30 * SR : PIECE), total);
      if (end < total) end = quietPoint(audio, end);
      const res = await t(audio.subarray(pos, end), { chunk_length_s: 30, stride_length_s: 5, return_timestamps: true });
      const offset = pos / SR;
      (res.chunks || [{ timestamp: [0], text: res.text }]).forEach((c) => {
        const text = c.text.trim();
        if (text) segments.push({ start: (c.timestamp[0] || 0) + offset, text });
      });
      pos = end;
      render(); output.scrollTop = output.scrollHeight;
      copyBtn.disabled = saveBtn.disabled = !segments.length;
      log("Finished " + fmt(pos / SR));
      const pct = (pos / total) * 100, elapsed = (performance.now() - t0) / 1000;
      setStatus(`Transcribing… ${Math.round(pct)}%. ${fmtEta((elapsed / pos) * (total - pos))} left. Keep this tab open.`, false, pct);
      heroBusy = true;
      await new Promise((r) => setTimeout(r));
    }
    if (!segments.length) output.textContent = "No speech was found in this file.";
    setStatus("Done.", false, 100);
    heroBusy = false;
  } catch (err) {
    console.error(err);
    log("Failed: " + (err.message || err));
    heroBusy = false;
    setStatus("Something went wrong: " + (err.message || err) + ". Try a smaller accuracy setting or turn off GPU mode.", false, 0, true);
  } finally { goBtn.disabled = false; wakeLock?.release?.(); }
});

// ---------- Output ----------
function fmt(sec) {
  const s = Math.floor(sec), m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
function render() {
  output.classList.remove("empty");
  output.classList.toggle("no-ts", !tsToggle.checked);
  output.innerHTML = segments.map((s) => `<p><time>${fmt(s.start)}</time>${escapeHtml(s.text)}</p>`).join("");
}
tsToggle.addEventListener("change", () => segments.length && render());

function plainText() {
  return segments.map((s) => (tsToggle.checked ? `[${fmt(s.start)}] ` : "") + s.text).join("\n");
}
copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(plainText());
  copyBtn.textContent = "Copied"; setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
});
saveBtn.addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([plainText()], { type: "text/plain" }));
  a.download = (currentFile?.name.replace(/\.[^.]+$/, "") || "transcript") + ".txt";
  a.click(); URL.revokeObjectURL(a.href);
});

// ---------- Helpers ----------
function setStatus(msg, indeterminate, pct = 0, isError = false) {
  statusBox.hidden = !msg;
  statusBox.classList.toggle("busy", !!indeterminate);
  heroBusy = !!indeterminate;
  statusBox.classList.toggle("error", isError);
  barFill.classList.toggle("indet", !!indeterminate);
  barFill.style.width = indeterminate ? "" : pct + "%";
  statusText.textContent = msg;
}
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

// ---------- Hero ribbon: gentle idle motion, livelier while working ----------
let heroBusy = false;
const hero = $("hero"), hctx = hero.getContext("2d");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
function sizeHero() {
  const r = devicePixelRatio || 1;
  hero.width = hero.clientWidth * r; hero.height = hero.clientHeight * r;
}
sizeHero(); addEventListener("resize", sizeHero);
function drawHero(t) {
  const w = hero.width, h = hero.height, gap = 9 * (devicePixelRatio || 1), bw = 4 * (devicePixelRatio || 1);
  hctx.clearRect(0, 0, w, h);
  const g = hctx.createLinearGradient(0, 0, w, 0);
  g.addColorStop(0, "#7c6cf0"); g.addColorStop(.55, "#f6a5c0"); g.addColorStop(1, "#ffd0a1");
  hctx.fillStyle = g;
  const speed = heroBusy ? 0.004 : 0.0011, amp = heroBusy ? 1 : 0.6;
  for (let x = 0, i = 0; x < w; x += gap, i++) {
    const n = Math.sin(i * 0.28 + t * speed) * 0.5 + Math.sin(i * 0.11 - t * speed * 1.7) * 0.5;
    const env = Math.sin((x / w) * Math.PI);
    const bar = Math.max(bw, (0.12 + Math.abs(n) * 0.88 * amp) * env * h);
    hctx.beginPath();
    hctx.roundRect(x, (h - bar) / 2, bw, bar, bw / 2);
    hctx.fill();
  }
  if (!reduceMotion) requestAnimationFrame(drawHero);
}
requestAnimationFrame(drawHero);

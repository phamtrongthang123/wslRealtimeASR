const statusEl = document.getElementById("status");
const sessionEl = document.getElementById("session");
const transcriptEl = document.getElementById("transcript");
const meterBar = document.getElementById("meterBar");
const noteEl = document.getElementById("note");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_MS = 500;
const CHUNK_SAMPLES = Math.round((TARGET_SAMPLE_RATE * CHUNK_MS) / 1000);

let audioContext;
let mediaStream;
let workletNode;
let sessionId;
let running = false;
let sendChain = Promise.resolve();
let lastLevel = 0;

function setStatus(text) {
  statusEl.textContent = text;
}

function setSession(text) {
  sessionEl.textContent = text;
}

function setTranscript(text) {
  transcriptEl.textContent = text || "";
}

function showNote(text) {
  noteEl.textContent = text;
}

async function createSession() {
  const resp = await fetch("/session", { method: "POST" });
  const data = await resp.json();
  if (!data.ok) {
    throw new Error(data.error || "Failed to create session");
  }
  return data.session_id;
}

async function stopSession(id) {
  const resp = await fetch("/stop", {
    method: "POST",
    headers: { "X-Session-Id": id },
  });
  const data = await resp.json();
  if (!data.ok) {
    throw new Error(data.error || "Failed to stop session");
  }
  return data.text || "";
}

function resampleBuffer(input, inputRate, targetRate) {
  if (inputRate === targetRate) {
    return input;
  }
  const ratio = inputRate / targetRate;
  const newLength = Math.round(input.length / ratio);
  const output = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const origin = i * ratio;
    const left = Math.floor(origin);
    const right = Math.min(left + 1, input.length - 1);
    const frac = origin - left;
    output[i] = input[left] * (1 - frac) + input[right] * frac;
  }
  return output;
}

function floatTo16BitPCM(float32Array) {
  const output = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

function updateMeter(samples) {
  if (!samples.length) return;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i];
  }
  const rms = Math.sqrt(sum / samples.length);
  const level = Math.min(1, rms * 4);
  lastLevel = lastLevel * 0.65 + level * 0.35;
  meterBar.style.width = `${(lastLevel * 100).toFixed(1)}%`;
}

async function sendChunk(buffer) {
  if (!sessionId) return;
  const resp = await fetch("/chunk", {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Session-Id": sessionId,
    },
    body: buffer,
  });
  const data = await resp.json();
  if (data.ok && data.text !== undefined) {
    setTranscript(data.text || "");
  }
}

function enqueueChunk(buffer) {
  sendChain = sendChain
    .then(() => sendChunk(buffer))
    .catch((err) => {
      console.error(err);
      showNote("Network error sending audio chunk. Check server logs.");
    });
}

async function startRecording() {
  if (running) return;
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  showNote("Recording live. Keep this tab open.");
  setStatus("Connecting...");
  setTranscript("");

  try {
    sessionId = await createSession();
    setSession(sessionId.slice(0, 8));

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    await audioContext.audioWorklet.addModule("/recorder-worklet.js");

    const source = audioContext.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioContext, "recorder-processor", {
      processorOptions: { chunkSize: CHUNK_SAMPLES },
    });

    const gainNode = audioContext.createGain();
    gainNode.gain.value = 0;

    source.connect(workletNode);
    workletNode.connect(gainNode).connect(audioContext.destination);

    workletNode.port.onmessage = (event) => {
      if (!running) return;
      const chunk = event.data;
      if (!(chunk instanceof Float32Array)) return;

      const resampled = resampleBuffer(chunk, audioContext.sampleRate, TARGET_SAMPLE_RATE);
      updateMeter(resampled);
      const pcm16 = floatTo16BitPCM(resampled);
      enqueueChunk(pcm16.buffer);
    };

    setStatus("Recording");
  } catch (err) {
    console.error(err);
    running = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("Error");
    showNote(err.message || "Failed to access microphone.");
  }
}

async function stopRecording() {
  if (!running) return;
  running = false;
  stopBtn.disabled = true;
  setStatus("Stopping...");

  if (workletNode) {
    workletNode.port.postMessage({ flush: true });
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  if (audioContext) {
    await audioContext.close();
  }

  try {
    await sendChain;
    const finalText = await stopSession(sessionId);
    setTranscript(finalText);
    setStatus("Idle");
    showNote("Session complete. You can start a new recording.");
  } catch (err) {
    console.error(err);
    setStatus("Error");
    showNote(err.message || "Failed to stop session.");
  } finally {
    startBtn.disabled = false;
    sessionId = null;
    setSession("Not started");
  }
}

startBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
clearBtn.addEventListener("click", () => {
  setTranscript("");
});

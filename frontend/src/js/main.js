// Config
// We point to the new FastAPI endpoints which are mounted under /api/v1/
const API_BASE = import.meta.env.VITE_API_URL || (location.hostname === 'localhost' ? 'http://localhost:8000/api/v1' : '/api/v1');
const AUDIO_UPLOAD_ENDPOINT = `${API_BASE}/audio`;
const SUMMARIZE_ENDPOINT = `${API_BASE}/summary`;
const PREDICT_SIGN_ENDPOINT = `${API_BASE}/sign`;
const FETCH_TIMEOUT_MS = 60_000; // 60 seconds default
const MAX_FILE_SIZE_MB = 50;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

// ---------- DOM ----------
const recordBtn = document.getElementById('record-btn');
const stopRecordBtn = document.getElementById('stop-record-btn');
const srBtn = document.getElementById('sr-btn');
const transcriptionOutput = document.getElementById('transcription-output');
const summaryOutput = document.getElementById('summary-output');
const summaryTypeSelect = document.getElementById('summary-type');
const extendedSummaryDetails = document.getElementById('extended-summary-details');
const summaryTitleText = document.getElementById('summary-title-text');
const keyPointsOutput = document.getElementById('key-points-output');
const actionItemsOutput = document.getElementById('action-items-output');
const keywordsOutput = document.getElementById('keywords-output');
const statusIndicator = document.getElementById('status-indicator');
const downloadTranscriptBtn = document.getElementById('download-transcript');
const downloadSummaryBtn = document.getElementById('download-summary');
const stopSpeechBtn = document.getElementById('stop-speech-btn');
const uploadLocalBtn = document.getElementById('upload-local');
const fileInput = document.getElementById('file-input');
const speakTranscriptBtn = document.getElementById('speak-transcript-btn');
const speakSummaryBtn = document.getElementById('speak-summary-btn');
const aiSummarizeBtn = document.getElementById('ai-summarize-btn');
const startSignBtn = document.getElementById('start-sign-btn');
const stopSignBtn = document.getElementById('stop-sign-btn');
const speakSignBtn = document.getElementById('speak-sign-btn');
const signVideo = document.getElementById('sign-video');
const signCanvas = document.getElementById('sign-canvas');
const signDetectedText = document.getElementById('sign-detected-text');
const signSection = document.querySelector('.sign-section');

// state
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isUsingBrowserSR = false;
let speechRecognition = null;
let fullTranscript = "";
let currentStream = null;
let hands = null;
let camera = null;
let isSignCapturing = false;
let signLandmarks = [];

// ---------- utilities ----------
function setStatus(text, listening=false){
  statusIndicator.textContent = text;
  statusIndicator.classList.toggle('status-listening', listening);
}

function setControlsDuringAction(active){
  recordBtn.disabled = active;
  srBtn.disabled = active;
  uploadLocalBtn.disabled = active;
  aiSummarizeBtn.disabled = active;
  speakTranscriptBtn.disabled = active;
  speakSummaryBtn.disabled = active;
}

function enableDownloads(){
  downloadTranscriptBtn.disabled = false;
  downloadSummaryBtn.disabled = false;
}

function downloadText(filename, text){
  const blob = new Blob([text], {type:'text/plain;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click();
  a.remove(); URL.revokeObjectURL(url);
}

async function fetchWithTimeout(url, options = {}, timeout = FETCH_TIMEOUT_MS, retryCount = 0){
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  options.signal = controller.signal;
  try{
    const res = await fetch(url, options);
    clearTimeout(id);
    if ((res.status === 502 || res.status === 503) && retryCount === 0) {
      const delay = Math.pow(2, retryCount) * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
      return fetchWithTimeout(url, options, timeout, retryCount + 1);
    }
    return res;
  }catch(e){
    clearTimeout(id);
    if (retryCount === 0 && (e.name === 'TypeError' || e.name === 'NetworkError')) {
      const delay = Math.pow(2, retryCount) * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
      return fetchWithTimeout(url, options, timeout, retryCount + 1);
    }
    throw e;
  }
}

function stopSpeaking(){
  if (window.speechSynthesis && window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }
}

// ---------- MediaRecorder flow ----------
async function startMediaRecording(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    currentStream = stream;
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      await uploadAudioBlob(blob);
    };
    mediaRecorder.start();
    isRecording = true;
    setStatus('Recording audio locally...', true);
    recordBtn.style.display = 'none';
    stopRecordBtn.style.display = 'inline-block';
    setControlsDuringAction(true);
  }catch(err){
    console.error('getUserMedia error', err);
    setStatus('Cannot access microphone. Check permissions.');
  }
}

function stopMediaRecording(){
  try{
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
      currentStream = null;
    }
  }catch(e){ console.warn(e) }
  isRecording = false;
  recordBtn.style.display = 'inline-block';
  stopRecordBtn.style.display = 'none';
  setControlsDuringAction(false);
  setStatus('Processing audio...');
}

// upload audio blob to backend
async function uploadAudioBlob(blob){
  if (blob.size > MAX_FILE_SIZE_BYTES) {
    setStatus(`File too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.`);
    return;
  }
  setStatus('Uploading and transcribing (server-side)...');
  setControlsDuringAction(true);
  stopSpeaking();
  transcriptionOutput.setAttribute('aria-busy', 'true');
  summaryOutput.setAttribute('aria-busy', 'true');
  try{
    const form = new FormData();
    form.append('file', blob, 'recording.webm');
    
    // NOTE: Make sure there's NO trailing slash here, since FastAPI is expecting `/api/v1/audio` not `/api/v1/audio/`
    const res = await fetchWithTimeout(AUDIO_UPLOAD_ENDPOINT, {
      method:'POST',
      body: form
    });
    if (!res.ok) {
      if (res.status === 413) throw new Error('File too large (server rejected)');
      if (res.status === 422) throw new Error('Validation error on server');
      throw new Error('Server responded ' + res.status);
    }
    const data = await res.json();
    const transcription = data.transcription || '';
    const summary = data.summary || '';
    
    const transPlaceholder = transcriptionOutput.querySelector('.placeholder');
    if (transPlaceholder) transPlaceholder.remove();
    const sumPlaceholder = summaryOutput.querySelector('.placeholder');
    if (sumPlaceholder) sumPlaceholder.remove();
    
    transcriptionOutput.textContent = transcription;
    summaryOutput.textContent = summary;
    setStatus('Done ✅');
    enableDownloads();
    speakTranscriptBtn.disabled = !transcription;
    speakSummaryBtn.disabled = !summary;
    aiSummarizeBtn.disabled = !transcription;
  }catch(err){
    console.error('uploadAudioBlob error', err);
    if (err.name === 'AbortError') setStatus('Request timed out.');
    else setStatus('Upload failed: ' + (err.message || 'Check server or network.'));
  }finally{
    setControlsDuringAction(false);
    transcriptionOutput.setAttribute('aria-busy', 'false');
    summaryOutput.setAttribute('aria-busy', 'false');
  }
}

// ---------- Browser SpeechRecognition fallback ----------
function initSpeechRecognition(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const rec = new SR();
  rec.lang = navigator.language || 'en-US';
  rec.interimResults = true;
  rec.continuous = true;

  let interim = '';
  fullTranscript = '';

  rec.onstart = () => { setStatus('Listening (browser SR)...', true); };
  rec.onresult = (event) => {
    interim = '';
    for (let i = event.resultIndex; i < event.results.length; i++){
      const txt = event.results[i][0].transcript;
      if (event.results[i].isFinal) fullTranscript += (fullTranscript ? ' ' : '') + txt;
      else interim += txt;
    }
    transcriptionOutput.textContent = (fullTranscript + ' ' + interim).trim() || '';
  };
  rec.onspeechstart = () => { setStatus('Listening...', true); };
  rec.onspeechend = () => {};
  rec.onerror = (e) => { console.warn('SR error', e); setStatus('Speech recognition error.'); };
  rec.onend = async () => {
    setControlsDuringAction(true);
    const finalText = (fullTranscript || '').trim();
    if (!finalText) { setStatus('No speech captured.'); setControlsDuringAction(false); return; }
    
    // We already have transcription, so let's just enable summary if we want
    const transPlaceholder = transcriptionOutput.querySelector('.placeholder');
    if (transPlaceholder) transPlaceholder.remove();
    
    transcriptionOutput.textContent = finalText;
    enableDownloads();
    speakTranscriptBtn.disabled = !finalText;
    aiSummarizeBtn.disabled = !finalText;
    setStatus('Done ✅');
    setControlsDuringAction(false);
  };
  return rec;
}

function startBrowserSR(){
  if (!speechRecognition) speechRecognition = initSpeechRecognition();
  if (!speechRecognition){ setStatus('Browser SpeechRecognition not supported.'); return; }
  try{
    speechRecognition.start();
    isUsingBrowserSR = true;
    srBtn.textContent = 'Stop Browser SR';
    srBtn.setAttribute('aria-pressed','true');
    setControlsDuringAction(true);
  }catch(e){ console.warn(e); setStatus('Failed to start speech recognition.'); }
}

function stopBrowserSR(){
  try{ if (speechRecognition) speechRecognition.stop(); }catch(e){ console.warn(e) }
  isUsingBrowserSR = false;
  srBtn.textContent = 'Use Browser Speech API';
  srBtn.setAttribute('aria-pressed','false');
}

// speak text
function speakText(text){
  try{
    stopSpeaking();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = navigator.language || 'en-US';
    window.speechSynthesis.speak(u);
    stopSpeechBtn.style.display = 'inline-block';
  }catch(e){ console.warn('speakText error', e) }
}
stopSpeechBtn.addEventListener('click', () => { stopSpeaking(); stopSpeechBtn.style.display = 'none'; });

// AI Summarize functionality
async function summarizeText(text) {
  if (!text || !text.trim()) {
    setStatus('No transcription to summarize.');
    return;
  }
  setStatus('Generating AI summary...');
  setControlsDuringAction(true);
  summaryOutput.setAttribute('aria-busy', 'true');
  try {
    const res = await fetchWithTimeout(SUMMARIZE_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
          text: text.trim(),
          summary_type: summaryTypeSelect ? summaryTypeSelect.value : "short"
      })
    });
    if (!res.ok) throw new Error('Server responded ' + res.status);
    
    const data = await res.json();
    const summary = data.summary || '';
    
    const placeholder = summaryOutput.querySelector('.placeholder');
    if (placeholder) placeholder.remove();
    
    summaryOutput.textContent = summary;

    if (data.title || (data.key_points && data.key_points.length) || (data.action_items && data.action_items.length) || (data.keywords && data.keywords.length)) {
        if(extendedSummaryDetails) extendedSummaryDetails.style.display = 'block';
        if(summaryTitleText) summaryTitleText.textContent = data.title || 'Summary Details';
        if(keyPointsOutput) keyPointsOutput.innerHTML = data.key_points && data.key_points.length ? '<strong>Key Points:</strong><br>' + data.key_points.map(k => `• ${k}`).join('<br>') : '';
        if(actionItemsOutput) actionItemsOutput.innerHTML = data.action_items && data.action_items.length ? '<strong>Action Items:</strong><br>' + data.action_items.map(a => `• ${a}`).join('<br>') : '';
        if(keywordsOutput) keywordsOutput.innerHTML = data.keywords && data.keywords.length ? '<strong>Keywords:</strong> ' + data.keywords.join(', ') : '';
    } else {
        if(extendedSummaryDetails) extendedSummaryDetails.style.display = 'none';
    }

    setStatus('Summary generated ✅');
    speakSummaryBtn.disabled = !summary;
    enableDownloads();
  } catch (err) {
    console.error('summarizeText error', err);
    setStatus('Summary failed: ' + (err.message || 'Check server or network.'));
  } finally {
    setControlsDuringAction(false);
    summaryOutput.setAttribute('aria-busy', 'false');
  }
}

aiSummarizeBtn.addEventListener('click', () => {
  const text = transcriptionOutput.textContent.trim();
  if (text) summarizeText(text);
});

speakTranscriptBtn.addEventListener('click', () => {
  const text = transcriptionOutput.textContent.trim();
  if (text) speakText(text);
});

speakSummaryBtn.addEventListener('click', () => {
  const text = summaryOutput.textContent.trim();
  if (text) speakText(text);
});

// Sign Language Recognition with MediaPipe
function initMediaPipeHands() {
  if (typeof Hands === 'undefined') {
    setStatus('MediaPipe Hands not loaded. Check network connection.');
    return null;
  }
  const handsInstance = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  handsInstance.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  handsInstance.onResults((results) => {
    const ctx = signCanvas.getContext('2d');
    ctx.save();
    ctx.clearRect(0, 0, signCanvas.width, signCanvas.height);
    ctx.drawImage(results.image, 0, 0, signCanvas.width, signCanvas.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
        drawLandmarks(ctx, landmarks, { color: '#FF0000', radius: 3 });
        
        if (results.multiHandLandmarks[0]) {
          signLandmarks = results.multiHandLandmarks[0];
          if (isSignCapturing) predictSign(signLandmarks);
        }
      }
    }
    ctx.restore();
  });
  return handsInstance;
}

function drawConnectors(ctx, points, connections, style) {
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.lineWidth;
  ctx.beginPath();
  for (const [start, end] of connections) {
    const point1 = points[start];
    const point2 = points[end];
    ctx.moveTo(point1.x * signCanvas.width, point1.y * signCanvas.height);
    ctx.lineTo(point2.x * signCanvas.width, point2.y * signCanvas.height);
  }
  ctx.stroke();
}

function drawLandmarks(ctx, points, style) {
  ctx.fillStyle = style.color;
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x * signCanvas.width, point.y * signCanvas.height, style.radius, 0, 2 * Math.PI);
    ctx.fill();
  }
}

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

let predictSignDebounce = null;
async function predictSign(landmarks) {
  if (predictSignDebounce) clearTimeout(predictSignDebounce);
  predictSignDebounce = setTimeout(async () => {
    if (!isSignCapturing || !landmarks || landmarks.length === 0) return;
    try {
      const res = await fetchWithTimeout(PREDICT_SIGN_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: landmarks.map(l => ({ x: l.x, y: l.y, z: l.z })) })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.text) {
          const placeholder = signDetectedText.querySelector('.placeholder');
          if (placeholder) placeholder.remove();
          signDetectedText.textContent = data.text;
          speakSignBtn.disabled = false;
        }
      }
    } catch (err) {
      console.error(err);
    }
  }, 500);
}

async function startSignCapture() {
  if (isSignCapturing) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    signVideo.srcObject = stream;
    signCanvas.width = 640;
    signCanvas.height = 480;
    
    if (!hands) {
      hands = initMediaPipeHands();
      if (!hands) { setStatus('Failed to initialize MediaPipe Hands.'); return; }
    }
    if (typeof Camera === 'undefined') { setStatus('MediaPipe Camera utils not loaded.'); return; }
    
    camera = new Camera(signVideo, {
      onFrame: async () => { await hands.send({ image: signVideo }); },
      width: 640, height: 480
    });
    
    camera.start();
    isSignCapturing = true;
    startSignBtn.style.display = 'none';
    stopSignBtn.style.display = 'inline-block';
    signSection.setAttribute('aria-busy', 'true');
    setStatus('Sign language capture active...');
  } catch (err) {
    console.error('startSignCapture error', err);
    setStatus('Failed to access camera. Check permissions.');
  }
}

function stopSignCapture() {
  if (!isSignCapturing) return;
  isSignCapturing = false;
  if (camera) { camera.stop(); camera = null; }
  if (signVideo.srcObject) {
    signVideo.srcObject.getTracks().forEach(track => track.stop());
    signVideo.srcObject = null;
  }
  const ctx = signCanvas.getContext('2d');
  ctx.clearRect(0, 0, signCanvas.width, signCanvas.height);
  
  startSignBtn.style.display = 'inline-block';
  stopSignBtn.style.display = 'none';
  signSection.setAttribute('aria-busy', 'false');
  setStatus('Sign capture stopped.');
}

startSignBtn.addEventListener('click', startSignCapture);
stopSignBtn.addEventListener('click', stopSignCapture);
speakSignBtn.addEventListener('click', () => {
  const text = signDetectedText.textContent.trim();
  if (text && !text.includes('placeholder')) speakText(text);
});

stopRecordBtn.addEventListener('click', stopMediaRecording);

recordBtn.addEventListener('click', async () => {
  if (isRecording) stopMediaRecording();
  else if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) await startMediaRecording();
  else setStatus('MediaRecorder not supported. Try browser SR.');
});

srBtn.addEventListener('click', () => {
  if (isUsingBrowserSR) stopBrowserSR();
  else startBrowserSR();
});

uploadLocalBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  setStatus('Uploading local file...');
  setControlsDuringAction(true);
  try{
    await uploadAudioBlob(file);
  }catch(e){
    console.error(e);
    setStatus('Upload failed.');
  }finally{
    setControlsDuringAction(false);
    fileInput.value = '';
  }
});

downloadTranscriptBtn.addEventListener('click', () => {
  const txt = transcriptionOutput.textContent.trim();
  if (!txt) return setStatus('Nothing to download.');
  downloadText('transcript.txt', txt);
});
downloadSummaryBtn.addEventListener('click', () => {
  const txt = summaryOutput.textContent.trim();
  if (!txt) return setStatus('Nothing to download.');
  downloadText('summary.txt', txt);
});

(function init(){
  if (!('mediaDevices' in navigator)) {
    setStatus('Note: MediaRecorder not available in this browser. Use the Browser Speech API or upload audio files.');
  } else {
    setStatus('Ready — use "Start Recording" to record audio or upload a file.');
  }
  recordBtn.addEventListener('keyup', (e) => { if (e.key === 'Enter') recordBtn.click(); });
  srBtn.addEventListener('keyup', (e) => { if (e.key === 'Enter') srBtn.click(); });
  stopSpeechBtn.style.display = 'none';
  stopRecordBtn.style.display = 'none';
  
  const obs = new MutationObserver(() => {
    const t = transcriptionOutput.textContent.trim();
    const s = summaryOutput.textContent.trim();
    if (t) downloadTranscriptBtn.disabled = false;
    if (s) downloadSummaryBtn.disabled = false;
  });
  obs.observe(transcriptionOutput, { childList:true, subtree:true, characterData:true });
  obs.observe(summaryOutput, { childList:true, subtree:true, characterData:true });
})();

window.addEventListener('beforeunload', () => {
  try { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); } catch(e){}
  try { if (speechRecognition) speechRecognition.stop(); } catch(e){}
  try { if (currentStream) currentStream.getTracks().forEach(track => track.stop()); } catch(e){}
  try { stopSignCapture(); } catch(e){}
});

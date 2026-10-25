let lastEmotion = "";
const statusEl = document.getElementById("status");
const xaiEl = document.getElementById("xai-text");
const video = document.getElementById("video");

// Reusable canvas (keeps the camera's aspect ratio so landmarks are not squashed)
const canvas = document.createElement("canvas");
canvas.width = 320;
canvas.height = 240;
const ctx = canvas.getContext("2d");

// ----------------------------------
// Start webcam
// ----------------------------------
async function startVideo() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    video.srcObject = stream;
    statusEl.innerText = "Camera ready…";
    detectLoop();
  } catch (err) {
    console.error("Camera error:", err);
    statusEl.innerText =
      "Cannot access the camera. Allow camera permission (use http://localhost:5000) and reload.";
  }
}

// ----------------------------------
// Emotion detection loop (next request starts only after the last finished)
// ----------------------------------
async function detectLoop() {
  try {
    if (video.videoWidth > 0) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL("image/jpeg", 0.8);

      const response = await fetch("/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
      });

      if (!response.ok) throw new Error("Server returned " + response.status);
      const data = await response.json();
      updateUI(data);
    }
  } catch (err) {
    console.error("Detection error:", err);
  }
  setTimeout(detectLoop, 1000);
}

function updateUI(data) {
  const { emotion, time, calibrating, face_detected, reason, progress, total } = data;

  // Status text
  if (!face_detected) {
    statusEl.innerText = "No face detected – look at the camera";
  } else if (calibrating) {
    statusEl.innerText =
      `Calibrating (${progress}/${total}) – keep a neutral face`;
  } else {
    statusEl.innerText = "Detected Emotion: " + emotion;
  }

  // XAI text
  xaiEl.innerHTML = `
    <b>Why this emotion?</b><br>
    The emotion <b>${emotion}</b> is selected only after repeated confirmation
    across multiple frames, reducing noise and random fluctuations.<br>
    <i>${reason}</i><br><br>

    <b>How was it decided?</b><br>
    A personalized neutral baseline is first calibrated, and emotions are
    detected based on deviations from this reference using facial landmarks.<br><br>

    <b>Time context:</b><br>
    Current time period is <b>${time}</b>, and content is aligned accordingly.
  `;

  // Load content only when the confirmed emotion changes
  if (!calibrating && emotion !== lastEmotion) {
    lastEmotion = emotion;
    loadRecommendations(emotion);
  }
}

// ----------------------------------
// Recalibrate button
// ----------------------------------
document.getElementById("recalibrate").addEventListener("click", async () => {
  await fetch("/reset", { method: "POST" });
  lastEmotion = "";
  statusEl.innerText = "Recalibrating – keep a neutral face";
});

// ----------------------------------
// Load recommendations
// ----------------------------------
function videoCards(items) {
  return items.map(v => `
    <div class="video-card">
      <a href="${v.url}" target="_blank" rel="noopener">
        <img src="${v.thumb}" alt="thumbnail">
        <p>${v.title}</p>
      </a>
    </div>
  `).join("");
}

async function loadRecommendations(emotion) {
  try {
    const res = await fetch(`/recommend/${emotion}`);
    const data = await res.json();

    document.getElementById("songs").innerHTML = videoCards(data.songs);
    document.getElementById("speeches").innerHTML = videoCards(data.speeches);

    // Spotify (embed only)
    document.getElementById("spotify").innerHTML = (data.spotify || []).map(link => `
      <iframe src="${link}" width="100%" height="152" frameborder="0"
              allow="encrypted-media"></iframe>
    `).join("");
  } catch (err) {
    console.error("Recommendation error:", err);
  }
}

startVideo();

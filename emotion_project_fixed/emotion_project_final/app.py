from flask import Flask, render_template, request, jsonify
import base64
import binascii
import datetime
import html
import json
import math
import os
import random
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
BASELINE_REQUIRED = 8          # frames (with a face) used to learn the neutral face
REQUIRED_FRAMES = 4            # consecutive frames needed to confirm an emotion

# ---------------------------------------------
# MediaPipe Face Mesh
# ---------------------------------------------
# NOTE: this uses the legacy `mp.solutions` API, which was removed from newer
# mediapipe releases.  requirements.txt pins a version that still has it.
if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "Your mediapipe version has no `mp.solutions` (legacy API removed). "
        "Run:  pip install -r requirements.txt   (needs Python 3.9-3.12)"
    )

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# FaceMesh and the global state below are not thread-safe, Flask serves
# requests in threads -> protect everything with one lock.
state_lock = threading.Lock()

# ---------------------------------------------
# Global State
# ---------------------------------------------
last_emotion = "neutral"
candidate_emotion = None
candidate_count = 0
baseline_frames = 0
baseline_sum = {}
last_reason = "Calibrating your neutral face"


def reset_state():
    """Forget the neutral baseline and start calibrating again."""
    global last_emotion, candidate_emotion, candidate_count
    global baseline_frames, baseline_sum, last_reason
    last_emotion = "neutral"
    candidate_emotion = None
    candidate_count = 0
    baseline_frames = 0
    baseline_sum = {}
    last_reason = "Calibrating your neutral face"


reset_state()

# ---------------------------------------------
# Time Context (XAI)
# ---------------------------------------------
def get_time_context():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

# ---------------------------------------------
# Feature extraction
# ---------------------------------------------
def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def extract_features(lm):
    """
    All features are divided by the face size, so they do not change when the
    person moves closer to / further from the camera.
    """
    face_w = _dist(lm[234], lm[454]) or 1e-6      # cheek to cheek
    face_h = _dist(lm[10], lm[152]) or 1e-6       # forehead to chin

    mouth_center_y = (lm[13].y + lm[14].y) / 2
    corners_y = (lm[61].y + lm[291].y) / 2

    # eyebrow (70 / 300) to upper eyelid (159 / 386)
    brow_gap = ((lm[159].y - lm[70].y) + (lm[386].y - lm[300].y)) / 2

    return {
        "mouth_width": _dist(lm[61], lm[291]) / face_w,
        "mouth_open": _dist(lm[13], lm[14]) / face_h,
        "corner_lift": (mouth_center_y - corners_y) / face_h,   # > 0 : corners up
        "brow_gap": brow_gap / face_h,                           # small : brows lowered
        "brow_dist": _dist(lm[107], lm[336]) / face_w,           # small : brows pulled together
    }

# ---------------------------------------------
# Emotion Detection Logic  (returns (emotion, reason))
# ---------------------------------------------
def detect_emotion_from_landmarks(lm):
    global baseline_frames, baseline_sum

    f = extract_features(lm)

    # CALIBRATION
    if baseline_frames < BASELINE_REQUIRED:
        for k, v in f.items():
            baseline_sum[k] = baseline_sum.get(k, 0.0) + v
        baseline_frames += 1
        return "neutral", "Calibrating your neutral face"

    base = {k: v / BASELINE_REQUIRED for k, v in baseline_sum.items()}

    width_ratio = f["mouth_width"] / base["mouth_width"]
    open_ratio = f["mouth_open"] / max(base["mouth_open"], 1e-6)
    lift_delta = f["corner_lift"] - base["corner_lift"]
    brow_ratio = f["brow_gap"] / base["brow_gap"]
    brow_dist_ratio = f["brow_dist"] / base["brow_dist"]

    # HAPPY: smile -> mouth corners go up and/or the mouth gets wider
    if lift_delta > 0.012 or width_ratio > 1.10:
        return "happy", "Mouth corners raised / mouth widened compared to your neutral face"

    # ANGRY: eyebrows lowered and pulled together
    if brow_ratio < 0.88 and brow_dist_ratio < 0.96:
        return "angry", "Eyebrows lowered and pulled together"

    # SAD: mouth corners drop, or eyebrows raised with a closed mouth
    if lift_delta < -0.008 or (brow_ratio > 1.10 and open_ratio < 1.15):
        return "sad", "Mouth corners down / eyebrows raised compared to your neutral face"

    return "neutral", "Face is close to your neutral baseline"

# ---------------------------------------------
# Content Data
# ---------------------------------------------
YOUTUBE_SONGS = {
    "happy": [
        ("Happy Telugu Songs", "https://www.youtube.com/watch?v=lDK9QqIzhwk"),
        ("Feel Good Telugu Music", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("Positive Vibes Music", "https://www.youtube.com/watch?v=3GwjfUFyY6M"),
        ("Joyful Mood Songs", "https://www.youtube.com/watch?v=OPf0YbXqDm0"),
        ("Smile Beats", "https://www.youtube.com/watch?v=ru0K8uYEZWw"),
    ],
    "sad": [
        ("Sad Telugu Songs", "https://www.youtube.com/watch?v=hoNb6HuNmU0"),
        ("Emotional Background Music", "https://www.youtube.com/watch?v=2Vv-BfVoq4g"),
        ("Lonely Mood Songs", "https://www.youtube.com/watch?v=RgKAFK5djSk"),
        ("Melancholic Music", "https://www.youtube.com/watch?v=7maJOI3QMu0"),
        ("Heart Touching Music", "https://www.youtube.com/watch?v=1ZYbU82GVz4"),
    ],
    "neutral": [
        ("Lo-fi Beats", "https://www.youtube.com/watch?v=jfKfPfyJRdk"),
        ("Chillhop Music", "https://www.youtube.com/watch?v=5qap5aO4i9A"),
        ("Focus Background Music", "https://www.youtube.com/watch?v=WPni755-Krg"),
        ("Study Music", "https://www.youtube.com/watch?v=kgx4WGK0oNU"),
        ("Ambient Sounds", "https://www.youtube.com/watch?v=DWcJFNfaw9c"),
    ],
    "angry": [
        ("Calm Music for Anger", "https://www.youtube.com/watch?v=lFcSrYw-ARY"),
        ("Peaceful Instrumentals", "https://www.youtube.com/watch?v=MIr3RsUWrdo"),
        ("Relaxing Nature Sounds", "https://www.youtube.com/watch?v=2OEL4P1Rz04"),
        ("Soothing Music", "https://www.youtube.com/watch?v=7maJOI3QMu0"),
        ("Mind Relaxation Music", "https://www.youtube.com/watch?v=1ZYbU82GVz4"),
    ],
}

YOUTUBE_SPEECHES = {
    "happy": [
        ("Positive Thinking Telugu Speech", "https://www.youtube.com/watch?v=ZXsQAXx_ao0"),
        ("Happiness Motivation Telugu", "https://www.youtube.com/watch?v=mgmVOuLgFB0"),
        ("Enjoy Life Telugu Speech", "https://www.youtube.com/watch?v=wnHW6o8WMas"),
        ("Success Mindset Telugu", "https://www.youtube.com/watch?v=KxGRhd_iWuE"),
        ("Believe Yourself Telugu", "https://www.youtube.com/watch?v=2zRT9I6W8Kk"),
    ],
    "sad": [
        ("You Are Not Alone Telugu Speech", "https://www.youtube.com/watch?v=wnHW6o8WMas"),
        ("Emotional Healing Telugu Talk", "https://www.youtube.com/watch?v=ZXsQAXx_ao0"),
        ("Motivation During Failure Telugu", "https://www.youtube.com/watch?v=mgmVOuLgFB0"),
        ("Mental Strength Telugu Speech", "https://www.youtube.com/watch?v=KxGRhd_iWuE"),
        ("Hope & Courage Telugu Talk", "https://www.youtube.com/watch?v=2zRT9I6W8Kk"),
    ],
    "neutral": [
        ("Daily Discipline Telugu Speech", "https://www.youtube.com/watch?v=mgmVOuLgFB0"),
        ("Focus Improvement Telugu Talk", "https://www.youtube.com/watch?v=KxGRhd_iWuE"),
        ("Self Development Telugu Speech", "https://www.youtube.com/watch?v=wnHW6o8WMas"),
        ("Life Advice Telugu Talk", "https://www.youtube.com/watch?v=ZXsQAXx_ao0"),
        ("Mind Control Telugu Speech", "https://www.youtube.com/watch?v=2zRT9I6W8Kk"),
    ],
    "angry": [
        ("Anger Management Telugu Speech", "https://www.youtube.com/watch?v=KxGRhd_iWuE"),
        ("Stay Calm Telugu Motivation", "https://www.youtube.com/watch?v=mgmVOuLgFB0"),
        ("Control Emotions Telugu Talk", "https://www.youtube.com/watch?v=ZXsQAXx_ao0"),
        ("Peaceful Mind Telugu Speech", "https://www.youtube.com/watch?v=wnHW6o8WMas"),
        ("Stress Control Telugu Talk", "https://www.youtube.com/watch?v=2zRT9I6W8Kk"),
    ],
}

SPOTIFY_LINKS = {
    "happy": [
        "https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC"
    ],
    "sad": [
    
        "https://open.spotify.com/embed/playlist/37i9dQZF1DX7qK8ma5wgG1"
    ],
    "neutral": [
        "https://open.spotify.com/embed/playlist/37i9dQZF1DX4WYpdgoIcn6"
    ],
    "angry": [
        "https://open.spotify.com/embed/playlist/37i9dQZF1DX0XUsuxWHRQd"
    ]
}


def format_youtube(items):
    return [{
        "title": html.escape(t),
        "url": u,
        "thumb": f"https://img.youtube.com/vi/{u.split('v=')[-1]}/0.jpg"
    } for t, u in items]


# ---------------------------------------------
# Live YouTube search (optional)
# ---------------------------------------------
# Put your key in an environment variable - never inside the code:
#   Windows (cmd):  set YOUTUBE_API_KEY=your_key
#   Mac / Linux  :  export YOUTUBE_API_KEY=your_key
# Without a key (or if YouTube fails / quota is over) the hard-coded lists
# above are used, so the app never breaks.
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "").strip()
CACHE_SECONDS = 3600            # a search costs 100 quota units (10,000/day free)

SEARCH_QUERIES = {
    "happy":   {"songs": "Telugu happy feel good songs",
                "speeches": "Telugu positive happiness motivational speech"},
    "sad":     {"songs": "Telugu sad emotional melody songs",
                "speeches": "Telugu motivational speech for sadness hope courage"},
    "neutral": {"songs": "Telugu lofi chill relaxing songs",
                "speeches": "Telugu self development focus motivational speech"},
    "angry":   {"songs": "Telugu calm relaxing peaceful music",
                "speeches": "Telugu anger management stay calm speech"},
}

_cache = {}                      # (emotion, kind) -> (timestamp, items)
_cache_lock = threading.Lock()


def youtube_search(query, max_results=5):
    """Return [(title, url), ...] from the YouTube Data API, or None on any problem."""
    if not YOUTUBE_API_KEY:
        return None
    params = urllib.parse.urlencode({
        "part": "snippet",
        "type": "video",
        "maxResults": max_results,
        "q": query,
        "regionCode": "IN",
        "relevanceLanguage": "te",
        "videoEmbeddable": "true",
        "safeSearch": "moderate",
        "key": YOUTUBE_API_KEY,
    })
    url = "https://www.googleapis.com/youtube/v3/search?" + params
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [
            (html.unescape(it["snippet"]["title"]),
             "https://www.youtube.com/watch?v=" + it["id"]["videoId"])
            for it in data.get("items", [])
            if it.get("id", {}).get("videoId")
        ] or None
    except Exception as exc:       # network error, bad key, quota exceeded ...
        print("YouTube API error:", exc)
        return None


def get_videos(emotion, kind):
    """Live results (cached) -> fallback to the hard-coded list."""
    fallback = (YOUTUBE_SONGS if kind == "songs" else YOUTUBE_SPEECHES).get(emotion, [])
    query = SEARCH_QUERIES.get(emotion, {}).get(kind)
    if not query:
        return fallback, "fallback"

    key = (emotion, kind)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1], "live"

    items = youtube_search(query)
    if items:
        with _cache_lock:
            _cache[key] = (time.time(), items)
        return items, "live"
    return random.sample(fallback, len(fallback)), "fallback"


# ---------------------------------------------
# Routes
# ---------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    global last_emotion, candidate_emotion, candidate_count, last_reason

    payload = request.get_json(silent=True) or {}
    data = payload.get("image", "")

    try:
        raw = base64.b64decode(data.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except (IndexError, binascii.Error, ValueError):
        img = None

    if img is None:
        return jsonify({"error": "invalid image"}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with state_lock:
        result = face_mesh.process(img_rgb)
        face_detected = bool(result.multi_face_landmarks)

        if face_detected:
            landmarks = result.multi_face_landmarks[0].landmark
            detected, reason = detect_emotion_from_landmarks(landmarks)

            if baseline_frames >= BASELINE_REQUIRED:
                if detected == candidate_emotion:
                    candidate_count += 1
                else:
                    candidate_emotion = detected
                    candidate_count = 1

                if candidate_count >= REQUIRED_FRAMES:
                    last_emotion = candidate_emotion
                    last_reason = reason

        response = {
            "emotion": last_emotion,
            "time": get_time_context(),
            "calibrating": baseline_frames < BASELINE_REQUIRED,
            "progress": min(baseline_frames, BASELINE_REQUIRED),
            "total": BASELINE_REQUIRED,
            "face_detected": face_detected,
            "reason": last_reason,
        }
    return jsonify(response)


@app.route("/reset", methods=["POST"])
def reset():
    with state_lock:
        reset_state()
    return jsonify({"ok": True})


@app.route("/recommend/<emotion>")
def recommend(emotion):
    songs, songs_src = get_videos(emotion, "songs")
    speeches, speeches_src = get_videos(emotion, "speeches")

    return jsonify({
        "songs": format_youtube(songs[:5]),
        "speeches": format_youtube(speeches[:5]),
        "spotify": SPOTIFY_LINKS.get(emotion, []),
        "source": "live" if "live" in (songs_src, speeches_src) else "fallback",
    })


# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)

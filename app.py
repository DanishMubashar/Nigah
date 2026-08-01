import os
import time
import asyncio
import threading
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO
import edge_tts
import io
import base64
import sounddevice as sd
import soundfile as sf
import random

# =========================================================
#                    LOAD ENVIRONMENT
# =========================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Settings
MODEL_FILE = "yolo11m.pt"
VOICE_ID = "ur-PK-UzmaNeural"
CONFIDENCE_THRESHOLD = 0.4

# =========================================================
#                     PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(
    page_title="Bina Rahnuma — Real-time Guide",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, #1b1f3b 0%, #0d0f1c 60%, #05060c 100%);
}

.hero {
    padding: 2.2rem 2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, #6C5CE7 0%, #00B4D8 100%);
    box-shadow: 0 10px 35px rgba(108, 92, 231, 0.35);
    margin-bottom: 1.8rem;
}
.hero h1 {
    color: white;
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #eaeaff;
    font-size: 1.02rem;
    margin: 0;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    backdrop-filter: blur(6px);
    margin-bottom: 1.2rem;
}
.card h3 {
    color: #ffffff;
    font-size: 1.05rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.guidance-box {
    background: linear-gradient(135deg, rgba(0,180,216,0.15), rgba(108,92,231,0.15));
    border: 1px solid rgba(0,180,216,0.4);
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: fadeIn 0.5s;
}
.guidance-text {
    font-family: 'Noto Nastaliq Urdu', serif;
    direction: rtl;
    font-size: 2rem;
    line-height: 2.6rem;
    color: #ffffff;
    font-weight: 700;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

.chip {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    background: rgba(0, 180, 216, 0.18);
    border: 1px solid rgba(0, 180, 216, 0.45);
    color: #d7f7ff;
    font-size: 0.85rem;
    margin: 0.2rem 0.3rem 0.2rem 0;
}
.chip.warn {
    background: rgba(255, 99, 99, 0.18);
    border: 1px solid rgba(255, 99, 99, 0.5);
    color: #ffd7d7;
}

.live-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: #00ff00;
    border-radius: 50%;
    animation: pulse 1s infinite;
    margin-right: 8px;
}

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
    100% { opacity: 1; transform: scale(1); }
}

.voice-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    background-color: #ff6b6b;
    border-radius: 50%;
    animation: pulse 0.5s infinite;
    margin-right: 8px;
}

.stButton>button {
    border-radius: 12px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    background: linear-gradient(120deg, #6C5CE7, #00B4D8);
    color: white;
    border: none;
}
.stButton>button:hover {
    opacity: 0.9;
    color: white;
}

section[data-testid="stSidebar"] {
    background: #10132a;
}

.guidance-history {
    max-height: 300px;
    overflow-y: auto;
    padding: 0.5rem;
}
.guidance-history::-webkit-scrollbar {
    width: 6px;
}
.guidance-history::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
}
.guidance-history::-webkit-scrollbar-thumb {
    background: #6C5CE7;
    border-radius: 10px;
}

.api-status {
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.8rem;
    margin: 5px 0;
}
.api-status.success {
    background: rgba(0, 255, 0, 0.1);
    color: #00ff00;
    border: 1px solid rgba(0, 255, 0, 0.2);
}
.api-status.error {
    background: rgba(255, 0, 0, 0.1);
    color: #ff6b6b;
    border: 1px solid rgba(255, 0, 0, 0.2);
}
.api-status.warning {
    background: rgba(255, 165, 0, 0.1);
    color: #ffa500;
    border: 1px solid rgba(255, 165, 0, 0.2);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
#                        SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🦯 Bina Rahnuma")
    st.markdown(
        """
        <div style="color:#b8bce0; font-size:0.9rem; line-height:1.6;">
        <b>Features:</b><br>
        • Image upload detection<br>
        • Real-time camera guide<br>
        • Instant voice feedback<br>
        • Urdu guidance<br>
        • <b>Auto-fallback to local AI</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("### 🎤 Voice Settings")
    voice_options = {
        "Urdu (Pakistan) - Uzma": "ur-PK-UzmaNeural",
        "Urdu (Pakistan) - Asad": "ur-PK-AsadNeural",
        "Arabic (Saudi) - Hoda": "ar-SA-HodaNeural",
        "English (US) - Jenny": "en-US-JennyNeural",
    }
    selected_voice = st.selectbox(
        "Select Voice",
        options=list(voice_options.keys()),
        index=0
    )
    VOICE_ID = voice_options[selected_voice]
    
    st.markdown("---")
    st.markdown("### 🤖 AI Mode")
    use_api = st.checkbox("Use Gemini API (if available)", value=True if GOOGLE_API_KEY else False)
    
    # Show API status
    if GOOGLE_API_KEY and use_api:
        st.markdown('<div class="api-status success">✅ Gemini API Available</div>', unsafe_allow_html=True)
    elif GOOGLE_API_KEY and not use_api:
        st.markdown('<div class="api-status warning">⚠️ Using Local AI (API disabled)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status error">❌ No API Key - Using Local AI</div>', unsafe_allow_html=True)

# =========================================================
#                    CACHED RESOURCES
# =========================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_name: str):
    return YOLO(model_name)

# =========================================================
#                    GEMINI API HANDLER
# =========================================================
def get_llm():
    """Initialize Gemini LLM with correct model name"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Correct model name format
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Changed from gemini-2.5-flash to gemini-1.5-flash
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return None

SYSTEM_PROMPT = """Aap ek madadgar assistant hain jo nabeena (blind) insaan ko chalte waqt real-time guidance dete hain.
Aapko camera se detect hone wali cheezon ki list (label, position, distance) di jayegi.

Aapka jawab:
- Sirf Urdu script mein ho (Urdu rasm-ul-khat), 1-2 chhoti lines se zyada nahi.
- Seedha aur actionable ho: batayein ke aage kya hai aur kya karna chahiye.
- Agar koi cheez qareeb aur seedhi samne ho to sabse pehle usay mention karein.
- Agar list khaali ho to sirf "راستہ صاف ہے، چلتے رہیں۔" ka mafhoom dein.
- Ghair zaroori tafseel na dein, seedha kaam ki baat karein.
"""

def generate_guidance_gemini(llm, detections: list) -> str:
    """Generate guidance using Gemini API"""
    if not detections:
        return "راستہ صاف ہے، چلتے رہیں۔"
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        detections_sorted = sorted(
            detections,
            key=lambda d: (d["distance"] != "bohat qareeb", d["distance"] != "qareeb"),
        )
        desc_lines = [f"- {d['label']} ({d['position']}, {d['distance']})" for d in detections_sorted[:3]]
        human_prompt = "Detect hone wali cheezein:\n" + "\n".join(desc_lines)
        
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])
        return response.content.strip()
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return None

# =========================================================
#                    LOCAL GUIDANCE GENERATOR
# =========================================================
def generate_guidance_local(detections: list) -> str:
    """
    Local guidance generator - NO API REQUIRED
    """
    if not detections:
        responses = [
            "راستہ صاف ہے، چلتے رہیں۔",
            "کوئی رکاوٹ نہیں، آگے بڑھیں۔",
            "سب صاف ہے، بلا خوف چلیں۔"
        ]
        return random.choice(responses)
    
    # Categorize objects
    dangerous = []
    nearby = []
    far = []
    
    for d in detections:
        if d["distance"] == "bohat qareeb":
            dangerous.append(d)
        elif d["distance"] == "qareeb":
            nearby.append(d)
        else:
            far.append(d)
    
    # Generate response
    if dangerous:
        objects = [f"{d['label']} ({d['position']})" for d in dangerous[:2]]
        if len(objects) == 1:
            return f"{objects[0]} بہت قریب ہے، فوراً رک جائیں!"
        else:
            return f"{' اور '.join(objects)} بہت قریب ہیں، رک جائیں!"
    
    elif nearby:
        objects = [f"{d['label']} ({d['position']})" for d in nearby[:2]]
        if len(objects) == 1:
            return f"{objects[0]} قریب ہے، آہستہ چلیں اور احتیاط کریں۔"
        else:
            return f"{' اور '.join(objects)} قریب ہیں، سست رفتاری سے چلیں۔"
    
    else:
        if far:
            obj = far[0]
            return f"{obj['label']} {obj['position']} میں دور ہے، چلتے رہیں۔"
        else:
            return random.choice(["راستہ صاف ہے، چلتے رہیں۔", "کوئی خطرہ نہیں، آگے بڑھیں۔"])

# =========================================================
#                    MAIN GUIDANCE FUNCTION
# =========================================================
def generate_guidance(llm, detections: list, use_api: bool) -> str:
    """Generate guidance using either API or local"""
    
    # Try API if enabled and available
    if use_api and GOOGLE_API_KEY and llm:
        try:
            result = generate_guidance_gemini(llm, detections)
            if result:
                return result
        except Exception as e:
            st.warning(f"API failed, using local AI: {e}")
    
    # Fallback to local
    return generate_guidance_local(detections)

# =========================================================
#                    CORE LOGIC FUNCTIONS
# =========================================================
def process_frame(frame, model, conf_thresh):
    """Process frame and return annotated frame with detections"""
    # Fix channel issue
    if len(frame.shape) == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    elif len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    results = model(frame, stream=False)
    annotated_frame = results[0].plot()
    detections = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            frame_h, frame_w = frame.shape[:2]
            x_center = (x1 + x2) / 2
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_w * frame_h
            area_ratio = box_area / frame_area
            
            # Position
            if x_center < frame_w / 3:
                position = "بائیں طرف"
            elif x_center > 2 * frame_w / 3:
                position = "دائیں طرف"
            else:
                position = "سیدھے سامنے"
            
            # Distance
            if area_ratio > 0.25:
                distance = "bohat qareeb"
            elif area_ratio > 0.08:
                distance = "qareeb"
            else:
                distance = "door"
            
            detections.append({
                "label": label,
                "position": position,
                "distance": distance,
                "confidence": round(conf, 2),
            })
    
    return annotated_frame, detections

# =========================================================
#                    TTS FUNCTIONS
# =========================================================
def text_to_speech(text: str, voice: str) -> bytes:
    """Generate TTS audio bytes"""
    try:
        async def generate_audio():
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        audio_bytes = asyncio.run(generate_audio())
        return audio_bytes
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def play_audio_simple(audio_bytes):
    """Simple audio playback"""
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        sd.play(audio_data, sample_rate)
        sd.wait()
        return True
    except Exception as e:
        print(f"Playback error: {e}")
        return False

def get_audio_html(audio_bytes, text):
    """Generate HTML audio player"""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio controls autoplay style="width: 100%; margin-top: 10px;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        return audio_html
    return ""

# =========================================================
#                          HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>🦯 Bina Rahnuma — Voice Guide System</h1>
        <p>Upload image or use camera • Instant Urdu voice guidance</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model
with st.spinner("Loading YOLO model..."):
    model = load_yolo_model(MODEL_FILE)

st.success("✅ Model loaded successfully!")

# Initialize LLM if API available
llm = None
if GOOGLE_API_KEY:
    try:
        llm = get_llm()
        if llm:
            st.success("✅ Gemini API initialized successfully!")
    except Exception as e:
        st.warning(f"⚠️ Gemini API init failed: {e}")

# =========================================================
#                    TAB FOR INPUT METHODS
# =========================================================
tab1, tab2 = st.tabs(["📸 Image Upload", "🎥 Real-time Camera"])

# =========================================================
#                    TAB 1: IMAGE UPLOAD
# =========================================================
with tab1:
    st.markdown("### Upload an image for detection")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp']
    )
    
    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(uploaded_file)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode == 'P':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            
            # Process image
            with st.spinner("Detecting objects..."):
                annotated_frame, detections = process_frame(
                    image_np, model, conf_thresh
                )
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(annotated_frame, caption="Detection Results", use_container_width=True)
                
                if detections:
                    chips_html = ""
                    for d in detections:
                        css_class = "chip warn" if d["distance"] in ("bohat qareeb", "qareeb") else "chip"
                        chips_html += (
                            f'<span class="{css_class}">{d["label"]} • {d["position"]} • '
                            f'{d["distance"]}</span>'
                        )
                    st.markdown(
                        f'<div class="card"><h3>📋 Detected Objects</h3>{chips_html}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No objects detected in this image.")
            
            with col2:
                # Generate guidance
                guidance_text = generate_guidance(llm, detections, use_api)
                
                # Display guidance
                ai_mode = "Gemini AI" if use_api and GOOGLE_API_KEY else "Local AI"
                st.markdown(
                    f"""
                    <div class="guidance-box">
                        <div style="color:#00B4D8; font-size:0.8rem; margin-bottom:0.5rem;">
                            🕐 Guidance ({ai_mode})
                        </div>
                        <div class="guidance-text">{guidance_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Generate voice
                with st.spinner("Generating voice..."):
                    audio_bytes = text_to_speech(guidance_text, VOICE_ID)
                    if audio_bytes:
                        if st.button("🔊 Play Voice", use_container_width=True):
                            threading.Thread(target=play_audio_simple, args=(audio_bytes,), daemon=True).start()
                            st.success("🔊 Playing voice guidance...")
                        
                        audio_html = get_audio_html(audio_bytes, guidance_text)
                        st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.error("Voice generation failed. Please try again.")
                        
        except Exception as e:
            st.error(f"Error processing image: {e}")

# =========================================================
#                    TAB 2: REAL-TIME CAMERA
# =========================================================
with tab2:
    st.markdown("### Real-time Camera Detection with Voice Guidance")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        start_btn = st.button("🎥 Start Camera", use_container_width=True)
        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)
    
    with col2:
        ai_mode = "Gemini AI" if use_api and GOOGLE_API_KEY else "Local AI"
        st.markdown(f"""
            <div style="color:#b8bce0; font-size:0.9rem; padding:10px;">
                <b>Instructions:</b><br>
                • Camera automatically detects objects<br>
                • Voice guidance plays automatically<br>
                • Object distance and position shown live<br>
                • <b>AI Mode:</b> {ai_mode}
            </div>
        """, unsafe_allow_html=True)
    
    # Session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'guidance_history' not in st.session_state:
        st.session_state.guidance_history = []
    
    if start_btn:
        st.session_state.camera_running = True
        st.session_state.guidance_history = []
    
    if stop_btn:
        st.session_state.camera_running = False
    
    if st.session_state.camera_running:
        # Create placeholders
        video_placeholder = st.empty()
        guidance_placeholder = st.empty()
        detections_placeholder = st.empty()
        history_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Status
        status_placeholder.markdown(
            """
            <div style="display:flex; align-items:center; gap:1rem;">
                <span class="live-indicator"></span>
                <span style="color:#00ff00; font-weight:600;">Camera Live</span>
                <span style="color:#b8bce0; margin-left:1rem;">|</span>
                <span style="color:#b8bce0;">🎤 Voice: </span>
                <span style="color:#00ff00;">Ready</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera open nahi ho pa raha.")
            st.session_state.camera_running = False
        else:
            last_process_time = time.time()
            last_guidance_text = ""
            FRAME_INTERVAL = 0.5
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                if current_time - last_process_time >= FRAME_INTERVAL:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated_frame, detections = process_frame(
                            frame_rgb, model, conf_thresh
                        )
                        
                        video_placeholder.image(annotated_frame, use_container_width=True)
                        
                        # Show detections
                        if detections:
                            chips_html = ""
                            for d in detections:
                                css_class = "chip warn" if d["distance"] in ("bohat qareeb", "qareeb") else "chip"
                                chips_html += (
                                    f'<span class="{css_class}">{d["label"]} • {d["position"]} • '
                                    f'{d["distance"]}</span>'
                                )
                            detections_placeholder.markdown(
                                f'<div class="card"><h3>📋 Live Detections</h3>{chips_html}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            detections_placeholder.markdown(
                                '<div class="card"><h3>📋 Live Detections</h3><span class="chip">✨ No objects</span></div>',
                                unsafe_allow_html=True
                            )
                        
                        # Generate guidance
                        guidance_text = generate_guidance(llm, detections, use_api)
                        
                        if guidance_text != last_guidance_text:
                            last_guidance_text = guidance_text
                            
                            # History
                            st.session_state.guidance_history.append({
                                'time': time.strftime('%H:%M:%S'),
                                'text': guidance_text,
                                'detections': len(detections)
                            })
                            if len(st.session_state.guidance_history) > 10:
                                st.session_state.guidance_history = st.session_state.guidance_history[-10:]
                            
                            # Display guidance
                            ai_mode = "Gemini AI" if use_api and GOOGLE_API_KEY else "Local AI"
                            guidance_placeholder.markdown(
                                f"""
                                <div class="guidance-box">
                                    <div style="color:#00B4D8; font-size:0.8rem; margin-bottom:0.5rem;">
                                        🕐 {time.strftime('%H:%M:%S')} ({ai_mode})
                                    </div>
                                    <div class="guidance-text">{guidance_text}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Update status
                            status_placeholder.markdown(
                                """
                                <div style="display:flex; align-items:center; gap:1rem;">
                                    <span class="live-indicator"></span>
                                    <span style="color:#00ff00; font-weight:600;">Camera Live</span>
                                    <span style="color:#b8bce0; margin-left:1rem;">|</span>
                                    <span class="voice-indicator"></span>
                                    <span style="color:#ff6b6b; font-weight:600;">Speaking...</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Generate and play voice
                            audio_bytes = text_to_speech(guidance_text, VOICE_ID)
                            if audio_bytes:
                                threading.Thread(target=play_audio_simple, args=(audio_bytes,), daemon=True).start()
                            
                            # Reset status
                            time.sleep(0.1)
                            status_placeholder.markdown(
                                """
                                <div style="display:flex; align-items:center; gap:1rem;">
                                    <span class="live-indicator"></span>
                                    <span style="color:#00ff00; font-weight:600;">Camera Live</span>
                                    <span style="color:#b8bce0; margin-left:1rem;">|</span>
                                    <span style="color:#b8bce0;">🎤 Voice: </span>
                                    <span style="color:#00ff00;">Ready</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Update history
                            history_html = ""
                            for entry in st.session_state.guidance_history[-5:]:
                                history_html += f"""
                                <div style="border-bottom:1px solid rgba(255,255,255,0.05); padding:0.5rem 0;">
                                    <span style="color:#5a5f8a; font-size:0.7rem;">{entry['time']}</span>
                                    <span style="color:#b8bce0; font-size:0.8rem;">| {entry['detections']} objects</span>
                                    <div style="font-family:'Noto Nastaliq Urdu',serif; direction:rtl; color:white; font-size:0.9rem; margin-top:0.2rem;">
                                        {entry['text']}
                                    </div>
                                </div>
                                """
                            history_placeholder.markdown(
                                f"""
                                <div class="card">
                                    <h3>📜 Guidance History</h3>
                                    <div class="guidance-history">
                                        {history_html}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        last_process_time = current_time
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                time.sleep(0.01)
            
            cap.release()
            st.session_state.camera_running = False
            
            video_placeholder.empty()
            guidance_placeholder.empty()
            detections_placeholder.empty()
            history_placeholder.empty()
            status_placeholder.markdown(
                """
                <div style="display:flex; align-items:center; gap:1rem;">
                    <span style="color:#ff6b6b;">⏹️</span>
                    <span style="color:#ff6b6b; font-weight:600;">Camera Stopped</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div class="card" style="text-align:center; padding: 3rem;">
                <h3 style="justify-content:center;">📷 Camera Off</h3>
                <p style="color:#b8bce0; font-size:1.1rem;">
                    "Start Camera" click karein aur real-time guidance shuru karein
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
#                   FOOTER
# =========================================================
st.markdown(
    """
    <div style="text-align:center; color:#5a5f8a; font-size:0.8rem; margin-top: 2rem; padding: 1rem;">
        Bina Rahnuma v4.0 • Gemini AI + Local AI • Free & Open Source
    </div>
    """,
    unsafe_allow_html=True
)
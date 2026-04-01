import streamlit as st
import requests
import io
from PIL import Image
import base64

# --- Page Setup ---
st.set_page_config(page_title="✨ Media Creator", layout="centered")

st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Media Creator")
st.caption("Endpoint: ylouolgkl7x17uss • AWS eu-west-1")

# --- CONFIGURATION ---
API_URL = "https://ylouolgkl7x17uss.eu-west-1.aws.endpoints.huggingface.cloud"

def process_response(response):
    """Processes binary data, base64 strings, or JSON errors."""
    ctype = response.headers.get("Content-Type", "")
    
    if response.status_code == 200:
        # Check for direct binary image or video
        if "image" in ctype or "video" in ctype or "octet-stream" in ctype:
            return response.content, "SUCCESS", ctype
        
        # Check for base64 string (some models return b64 inside JSON)
        try:
            text_data = response.text.strip().strip('"').strip("'")
            if text_data.startswith("iVBORw0") or text_data.startswith("/9j/"):
                return base64.b64decode(text_data), "SUCCESS", "image/png"
            # Video base64 header check (mp4)
            if text_data.startswith("AAAAIGZ0eX"):
                return base64.b64decode(text_data), "SUCCESS", "video/mp4"
            
            return None, f"Unexpected data format: {text_data[:100]}", None
        except:
            return None, "Failed to parse response body", None
    
    return None, f"Error {response.status_code}: {response.text}", None

def generate(prompt_text, mode):
    # Standard payload
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "negative_prompt": "blurry, low quality, distorted, text, watermark, anime, cartoon",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }
    }
    
    # If Video mode is selected, add temporal parameters
    if mode == "Video":
        # Note: These parameters only work if your HF Endpoint is running a Video model (like SVD or AnimateDiff)
        payload["parameters"]["num_frames"] = 16 
        payload["parameters"]["fps"] = 8
    
    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        return process_response(response)
    except Exception as e:
        return None, f"Connection Failed: {str(e)}", None

# --- UI ---
default_adiyogi = (
    "Cinematic photorealistic shot of Adiyogi Shiva statue, majestic black steel, "
    "crescent moon in hair, third eye, Velliangiri mountains background, "
    "night time, atmospheric lighting, hyper-detailed, 8k resolution"
)

prompt = st.text_area("Scene Description", value=default_adiyogi, height=120)

# Mode Selector
gen_mode = st.radio("Select Output Format", ["Image", "Video"], horizontal=True)

if st.button(f"✨ Generate {gen_mode}"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner(f"Creating your {gen_mode.lower()}..."):
            data, status, content_type = generate(prompt, gen_mode)
            
            if status == "SUCCESS":
                if "video" in content_type or gen_mode == "Video":
                    try:
                        st.video(data)
                        st.download_button("⬇️ Download Video", data, "output.mp4", "video/mp4")
                    except:
                        st.error("Model returned data that couldn't be played as video. It might still be returning images.")
                else:
                    image = Image.open(io.BytesIO(data))
                    st.image(image, use_container_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "output.png", "image/png")
            else:
                st.error("Server Message:")
                st.code(status)

st.divider()
st.info("**Requirement:** For 'Video' to work, your Hugging Face Endpoint must be configured with a **Text-to-Video** task. If it is a Text-to-Image endpoint, it will ignore the video parameters and return a still image.")
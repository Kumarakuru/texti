import streamlit as st
import requests
import io
from PIL import Image
from huggingface_hub import InferenceClient
import os
import traceback  # Add this at the top
import json

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
st.caption("Image: Dedicated Endpoint • Video: Inference Providers")

# --- CONFIGURATION ---
IMAGE_API_URL = "https://ylouolgkl7x17uss.eu-west-1.aws.endpoints.huggingface.cloud"

VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B"
VIDEO_PROVIDER = "fal-ai"

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

def process_image_response(response):
    if response.status_code == 200:
        ctype = response.headers.get("Content-Type", "")
        if "image" in ctype or "octet-stream" in ctype:
            return response.content, "SUCCESS"
    return None, f"Error {response.status_code}: {response.text[:300]}"

def generate_image(prompt_text):
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
    try:
        response = requests.post(IMAGE_API_URL, json=payload, timeout=180)
        return process_image_response(response)
    except Exception as e:
        return None, f"Connection Failed: {str(e)}"

# --- Updated Configuration ---
 # --- Configuration ---
# --- Stable Config ---
VIDEO_MODEL = "tencent/HunyuanVideo"

def generate_video(prompt_text):
    if not HF_TOKEN:
        return None, "HF_TOKEN is missing."

    # This is the canonical 2026 Router URL
    API_URL = f"https://api-inference.huggingface.co/models/{VIDEO_MODEL}"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "x-use-cache": "false",
        "x-wait-for-model": "true"
    }
    
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "num_frames": 16,
            "fps": 8
        },
        # 'auto' lets the HF Router pick the provider that ISN'T broken
        "provider": "auto" 
    }

    try:
        with st.spinner("🎥 Router is finding a provider and generating..."):
            response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
            
            if response.status_code != 200:
                return None, f"Router Error {response.status_code}: {response.text}"

            result = response.json()
            
            # Defensive parsing for the video URL
            video_url = None
            if isinstance(result, dict):
                video_url = result.get("url") or result.get("video", {}).get("url") or result.get("output")
            elif isinstance(result, list) and len(result) > 0:
                video_url = result[0].get("url") if isinstance(result[0], dict) else result[0]

            if video_url and isinstance(video_url, str) and video_url.startswith("http"):
                video_bytes = requests.get(video_url, timeout=60).content
                return video_bytes, "SUCCESS"
            
            return None, f"Task finished but could not find URL. Response: {result}"

    except Exception as e:
        import traceback
        return None, traceback.format_exc()



# --- UI ---
default_prompt = "A cute cat meowing softly in a cozy room, realistic, detailed fur, warm lighting"

prompt = st.text_area("Scene Description", value=default_prompt, height=120)

gen_mode = st.radio("Select Output Format", ["Image", "Video"], horizontal=True)

if st.button(f"✨ Generate {gen_mode}"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        if gen_mode == "Image":
            with st.spinner("Generating image..."):
                data, status = generate_image(prompt)
                if status == "SUCCESS":
                    image = Image.open(io.BytesIO(data))
                    st.image(image, use_container_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "output.png", "image/png")
                else:
                    st.error(status)
        else:  # Video
            if not HF_TOKEN:
                st.error("HF_TOKEN is required for video generation.")
            else:
                video_bytes, status = generate_video(prompt)
                
                if status == "SUCCESS" and video_bytes:
                    st.video(video_bytes)
                    st.download_button(
                        "⬇️ Download Video", 
                        video_bytes, 
                        "generated_video.mp4", 
                        "video/mp4"
                    )
                else:                                        
                    st.error("Video Generation Failed")
                    st.code(status, language="python") # This will display the full traceback in a scrollable box

st.divider()
st.info("Video uses pay-per-use Inference Providers and takes longer than images.")
import streamlit as st
import requests
import io
from PIL import Image
from huggingface_hub import InferenceClient
import os

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
st.caption("Image: Dedicated Endpoint • Video: Inference Providers (fal-ai)")

# --- CONFIGURATION ---
IMAGE_API_URL = "https://ylouolgkl7x17uss.eu-west-1.aws.endpoints.huggingface.cloud"

# Video Configuration - Changed to a proper text-to-video model
VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B"     # Good quality + supported for text-to-video
VIDEO_PROVIDER = "fal-ai"

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

def process_image_response(response):
    ctype = response.headers.get("Content-Type", "")
    if response.status_code == 200:
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

def generate_video(prompt_text):
    if not HF_TOKEN:
        return None, "❌ HF_TOKEN is missing. Add it in Streamlit secrets or as environment variable."

    client = InferenceClient(provider=VIDEO_PROVIDER, api_key=HF_TOKEN)

    try:
        with st.spinner("🎥 Generating video... (this usually takes 40–120 seconds)"):
            video_bytes = client.text_to_video(
                prompt=prompt_text,
                model=VIDEO_MODEL,
                extra_body={
                    "num_frames": 16,   # Short video (~2 seconds at 8 fps)
                    "fps": 8,
                }
            )
        return video_bytes, "SUCCESS"
    except Exception as e:
        error_str = str(e).lower()
        # Auto fallback if provider/model issue
        if "not supported" in error_str or "fal-ai" in error_str:
            try:
                st.info("Trying with auto provider selection...")
                client = InferenceClient(provider="auto", api_key=HF_TOKEN)
                video_bytes = client.text_to_video(prompt=prompt_text, model=VIDEO_MODEL)
                return video_bytes, "SUCCESS (auto provider)"
            except Exception as e2:
                return None, f"Failed with auto provider: {str(e2)}"
        return None, f"Video generation failed: {str(e)}"

# --- UI ---
default_prompt = (
    "Cinematic photorealistic shot of Adiyogi Shiva statue, majestic black steel, "
    "crescent moon in hair, third eye, Velliangiri mountains background, "
    "night time, atmospheric lighting, hyper-detailed, 8k resolution"
)

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

        else:  # Video mode
            if not HF_TOKEN:
                st.error("HF_TOKEN is required for video generation.")
                st.info("Create a token at https://huggingface.co/settings/tokens (with inference permission) and add it as secret `HF_TOKEN`")
            else:
                video_bytes, status = generate_video(prompt)
                
                if status.startswith("SUCCESS"):
                    st.video(video_bytes)
                    st.download_button(
                        "⬇️ Download Video", 
                        video_bytes, 
                        "generated_video.mp4", 
                        "video/mp4"
                    )
                else:
                    st.error(status)

st.divider()
st.info("""
**Current Setup:**
- **Image**: Your dedicated fast endpoint
- **Video**: Wan-AI/Wan2.2-TI2V-5B via fal-ai (proper text-to-video model)

**Tips:**
- Start with short, simple prompts to save credits.
- Video generation is slower and more expensive than images.
- If you still get errors, try changing `VIDEO_MODEL` to `"tencent/HunyuanVideo"` or `"Wan-AI/Wan2.1-T2V-14B"`.
""")
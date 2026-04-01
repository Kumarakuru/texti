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
        # Check for direct binary image
        if "image" in ctype or "video" in ctype:
            return response.content, "SUCCESS"
        
        # Check for base64 string in response body
        text_data = response.text.strip().strip('"').strip("'")
        if text_data.startswith("iVBORw0") or text_data.startswith("/9j/"):
            return base64.b64decode(text_data), "SUCCESS"
            
        return None, f"Unexpected data format: {text_data[:100]}"
    
    return None, f"Error {response.status_code}: {response.text}"

def generate(prompt_text):
    # Optimized for SD v1.5 or similar architectures
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
        response = requests.post(API_URL, json=payload, timeout=120)
        return process_response(response)
    except Exception as e:
        return None, f"Connection Failed: {str(e)}"

# --- UI ---
default_adiyogi = (
    "Cinematic photorealistic shot of Adiyogi Shiva statue, majestic black steel, "
    "crescent moon in hair, third eye, Velliangiri mountains background, "
    "night time, atmospheric lighting, hyper-detailed, 8k resolution"
)

prompt = st.text_area("Scene Description", value=default_adiyogi, height=120)

if st.button("✨ Generate"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("🎨 Creating..."):
            data, status = generate(prompt)
            
            if status == "SUCCESS":
                try:
                    image = Image.open(io.BytesIO(data))
                    st.image(image, use_container_width=True)
                    
                    # Download option
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "output.png", "image/png")
                except:
                    # If it's not an image, it might be a video file (mp4)
                    st.video(data)
                    st.download_button("⬇️ Download Video", data, "output.mp4", "video/mp4")
            else:
                st.error("Server Message:")
                st.code(status)

st.divider()
st.caption("Tip: If the result is a still image, ensure your endpoint task is set to 'Text-to-Video'.")
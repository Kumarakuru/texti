import streamlit as st
import requests
import io
from PIL import Image
import base64

# --- Page Setup ---
st.set_page_config(page_title="✨ Magic Image Creator", layout="centered")

st.markdown("""
<style>
    .main {background-color: #0f0f23;}
    h1 {color: #ff9ff3;}
    .stButton>button {
        background: linear-gradient(45deg, #ff9ff3, #f368e0);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-weight: bold;
        width: 100%;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Magic Image Creator")
st.caption("Model: Animagine XL 2.0 • Public Endpoint")

# --- CONFIGURATION ---
API_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"

def process_response(response):
    """Handles both raw binary and base64 string responses."""
    content_type = response.headers.get("Content-Type", "")
    
    # 1. Check if it's already raw bytes (image/png etc)
    if "image" in content_type:
        return response.content, None

    # 2. Check if it's a base64 string inside JSON or plain text
    try:
        text_data = response.text
        # If it looks like base64 (starts with iVBOR...), decode it
        if text_data.startswith("iVBORw0") or len(text_data) > 1000:
            return base64.b64decode(text_data), None
        
        # If it's actual JSON error
        return None, response.json()
    except Exception as e:
        return None, f"Status {response.status_code}: {response.text}"

def generate_image(prompt_text):
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "negative_prompt": "lowres, bad anatomy, bad hands, text, error, cropped, low quality",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024
        }
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=90)
        return process_response(response)
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# --- UI ---
# Added initial value as requested
prompt = st.text_area(
    "Describe your anime-style image ✨",
    value="1girl, solo, long hair, looking at viewer, masterpiece, high quality, cinematic lighting",
    height=120
)

if st.button("✨ Generate Image"):
    if not prompt.strip():
        st.error("Please enter a description.")
    else:
        with st.spinner("🎨 Creating..."):
            image_bytes, error = generate_image(prompt)
            
            if error:
                st.error("Server Error Detail:")
                st.code(error) 
            else:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    st.success("✨ Success!")
                    st.image(image, use_container_width=True)
                    
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "animagine.png", "image/png")
                except Exception as e:
                    st.error(f"Render Error: {e}")
                    st.write("First 100 chars of data:", str(image_bytes)[:100])

st.divider()
st.caption("Powered by Hugging Face Dedicated Endpoints")
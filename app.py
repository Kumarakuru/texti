import streamlit as st
import requests
import io
from PIL import Image
import base64
import json

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
        height: 3.5em;
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
    """Detects if response is image bytes, base64 string, or an error."""
    content_type = response.headers.get("Content-Type", "")
    
    # 1. Successful Binary Image
    if response.status_code == 200 and "image" in content_type:
        return response.content, None

    # 2. Potential Base64 String or JSON Error
    try:
        text_data = response.text
        # Check if the text is actually a Base64 image (common in some HF setups)
        if text_data.startswith("iVBORw0") or text_data.startswith("/9j/"):
            return base64.b64decode(text_data), None
        
        # Try parsing as JSON to see if there's a specific error message
        error_json = response.json()
        return None, error_json
    except:
        # Fallback to raw text if it's not JSON or Base64
        return None, f"Status {response.status_code}: {response.text}"

def generate_image(prompt_text):
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "negative_prompt": "shirt, clothes, human, lowres, bad anatomy, text, error, cropped, worst quality, low quality",
            "num_inference_steps": 35,
            "guidance_scale": 9.0,
            "width": 1024,
            "height": 1024
        }
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        return process_response(response)
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# --- UI ---
# Optimized prompt for Adiyogi using descriptive tags for the anime model
default_adiyogi = (
    "adiyogi shiva statue, large black stone bust, crescent moon in hair, "
    "serene face, third eye, meditation, mountains background, "
    "masterpiece, high quality, cinematic lighting, night sky, stars"
)

prompt = st.text_area(
    "Describe your image ✨",
    value=default_adiyogi,
    height=120
)

if st.button("✨ Generate Image"):
    if not prompt.strip():
        st.error("Please enter a description.")
    else:
        with st.spinner("🎨 Connecting to Hugging Face..."):
            image_data, error = generate_image(prompt)
            
            if error:
                st.error("Server Error/Response:")
                # Prints the exact error from the server (JSON or Text)
                st.code(error)
                if "loading" in str(error).lower():
                    st.info("The model is still loading on the GPU. Please try again in 1 minute.")
            else:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    st.success("✨ Success!")
                    st.image(image, use_container_width=True)
                    
                    # Download
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "adiyogi.png", "image/png")
                except Exception as e:
                    st.error(f"Failed to process result: {e}")
                    # Debugging: show what was actually received
                    st.text("Snippet of received data:")
                    st.write(str(image_data)[:200])

st.divider()
st.caption("Tip: Use tags like 'black stone bust' or 'crescent moon' to help the model identify Adiyogi.")
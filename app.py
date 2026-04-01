import streamlit as st
import requests
import io
from PIL import Image
import base64
import re

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

def is_base64(s):
    """Check if a string is likely a base64 encoded image."""
    return bool(re.match(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", s))

def process_response(response):
    content_type = response.headers.get("Content-Type", "")
    
    # 1. Direct Binary Image
    if "image" in content_type:
        return response.content, None

    # 2. Handle the String Issue
    text_data = response.text.strip()
    
    # Check if string starts with standard PNG/JPEG base64 headers or looks like a long b64 block
    if text_data.startswith("iVBORw0") or text_data.startswith("/9j/") or len(text_data) > 10000:
        try:
            # Clean possible quotes if it's a JSON-wrapped string
            clean_b64 = text_data.strip('"').strip("'")
            return base64.b64decode(clean_b64), None
        except Exception as e:
            return None, f"Base64 Decode Error: {str(e)}"

    # 3. Actual Error JSON or Text
    try:
        return None, response.json()
    except:
        return None, text_data

def generate_image(prompt_text):
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "negative_prompt": "shirt, clothes, human, lowres, bad anatomy, text, error, cropped, low quality",
            "num_inference_steps": 35,
            "guidance_scale": 9.0,
            "width": 1024,
            "height": 1024
        }
    }
    
    try:
        # Increased timeout for SDXL generation
        response = requests.post(API_URL, json=payload, timeout=120)
        return process_response(response)
    except Exception as e:
        return None, f"Network Error: {str(e)}"

# --- UI ---
default_prompt = (
    "adiyogi shiva statue, large black stone bust, crescent moon in hair, "
    "serene face, third eye, meditation, mountains background, "
    "masterpiece, high quality, cinematic lighting, night sky, stars"
)

prompt = st.text_area("Describe your image ✨", value=default_prompt, height=120)

if st.button("✨ Generate Image"):
    if not prompt.strip():
        st.error("Please enter a description.")
    else:
        with st.spinner("🎨 Processing image data..."):
            image_data, error = generate_image(prompt)
            
            if error:
                # If we get that iVBORw0 string here, it means decoding failed
                st.error("Server Response (Parsed as Error):")
                st.code(str(error)[:1000] + "...") 
            else:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    st.success("✨ Success!")
                    st.image(image, use_container_width=True)
                    
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "adiyogi_gen.png", "image/png")
                except Exception as e:
                    st.error(f"Render Error: {e}")
                    st.text("Raw data start:")
                    st.write(str(image_data)[:100])

st.divider()
st.caption("Tip: If the prompt doesn't look like Adiyogi, try adding 'Lord Shiva' or 'Yoga posture'.")
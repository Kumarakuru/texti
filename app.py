import streamlit as st
import requests
import io
from PIL import Image
import json

# --- Page Config ---
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
st.caption("Public Mode • Direct Inference")

# --- CONFIGURATION ---
API_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"

def query_hf_endpoint(prompt_text, steps):
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "num_inference_steps": steps,
            "guidance_scale": 7.5
        }
    }
    
    # Direct POST
    response = requests.post(API_URL, json=payload)
    
    # Check if the response is actually an image
    content_type = response.headers.get("Content-Type", "")
    
    if response.status_code == 200 and "image" in content_type:
        return response.content, None
    else:
        # If not an image, it's likely a JSON error or plain text
        try:
            error_data = response.json()
            error_msg = error_data.get("error", str(error_data))
        except:
            error_msg = response.text
            
        return None, f"Status {response.status_code}: {error_msg}"

# --- UI ---
prompt = st.text_area("Describe your image ✨", placeholder="A neon cat in Tokyo...", height=100)
steps = st.slider("Quality (Steps)", 10, 50, 30)

if st.button("✨ Generate Image"):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        with st.spinner("🎨 Connecting..."):
            image_bytes, error = query_hf_endpoint(prompt, steps)
            
            if error:
                st.error("Endpoint returned an error instead of an image:")
                st.code(error) # Shows the actual error from the HF server
                if "loading" in error.lower():
                    st.info("The model is currently loading into GPU memory. Please wait a minute and try again.")
            else:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    st.success("✨ Success!")
                    st.image(image, use_container_width=True)
                    
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download PNG", buf.getvalue(), "magic.png", "image/png")
                except Exception as e:
                    st.error(f"Image processing failed: {e}")
                    st.write("Raw response snippet:", image_bytes[:100]) # Debug what was sent

st.divider()
st.caption("Tip: If you see 'Model Loading', the endpoint just needs a moment to warm up.")
import streamlit as st
import requests
import io
from PIL import Image

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
# Using the exact URL from your screenshot
API_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"

def generate_image(prompt_text):
    # Animagine XL / SDXL usually expects this structure
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 1024,
            "height": 1024
        }
    }
    
    response = requests.post(API_URL, json=payload)
    
    # Check if the response is an image or an error
    if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
        return response.content, None
    else:
        try:
            return None, response.json()
        except:
            return None, response.text

# --- UI ---
prompt = st.text_area(
    "Describe your anime-style image ✨",
    placeholder="1girl, solo, long hair, looking at viewer, masterpiece, high quality, cinematic lighting",
    height=120
)

if st.button("✨ Generate Image"):
    if not prompt.strip():
        st.error("Please enter a description.")
    else:
        with st.spinner("🎨 Creating your masterpiece..."):
            image_data, error = generate_image(prompt)
            
            if error:
                st.error("Endpoint Error:")
                st.code(error)
                if "loading" in str(error).lower():
                    st.info("The model is currently loading into GPU memory. Try again in 60 seconds.")
            else:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    st.success("✨ Success!")
                    st.image(image, use_container_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Image",
                        data=buf.getvalue(),
                        file_name="animagine_gen.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Render Error: {e}")

st.divider()
st.caption("Powered by Hugging Face Dedicated Endpoints")
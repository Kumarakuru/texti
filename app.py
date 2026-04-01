import streamlit as st
from openai import OpenAI
import time
from PIL import Image
import io

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
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Magic Image Creator")
st.caption("Text to Image • Powered by Hugging Face")

# Your Endpoint
IMAGE_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud/v1"
client = OpenAI(base_url=IMAGE_URL.rstrip('/'), api_key="hf_dummy")

# Warmup Helper (same as your Shopify app)
def wait_for_hf_endpoint(fn, label="Image API", max_wait=180, interval=25):
    start = time.time()
    attempt = 0
    while True:
        try:
            result = fn()
            if attempt > 0:
                st.success(f"✅ {label} is ready!")
            return result, None
        except Exception as e:
            elapsed = int(time.time() - start)
            if "503" not in str(e):
                return None, f"Error: {str(e)}"
            if elapsed >= max_wait:
                return None, "Endpoint did not wake up in time."
            attempt += 1
            st.warning(f"⏳ {label} warming up... (attempt {attempt})")
            time.sleep(interval)

# UI
prompt = st.text_area("Describe your image", 
    placeholder="A cute baby dragon eating ice cream in a cyberpunk city at night, vibrant colors",
    height=110)

col1, col2 = st.columns(2)
with col1:
    num_images = st.slider("Number of images", 1, 4, 2)
with col2:
    steps = st.slider("Quality (Steps)", 20, 50, 30)

if st.button("✨ Generate Images", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please write a description!")
    else:
        with st.spinner("Creating your magic images..."):
            def generate():
                return client.images.generate(
                    model="black-forest-labs/FLUX.1-schnell",   # change if your endpoint uses different model
                    prompt=prompt,
                    num_images=num_images,
                    num_inference_steps=steps,
                    response_format="b64_json"
                )

            response, error = wait_for_hf_endpoint(generate, label="Image Generation")

            if error:
                st.error(error)
            else:
                st.success("Here are your images!")
                for i, img in enumerate(response.data):
                    image = Image.open(io.BytesIO(img.b64_json.encode()))
                    st.image(image, caption=f"Image {i+1}", use_column_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(f"⬇️ Download Image {i+1}", buf.getvalue(), f"image_{i+1}.png", mime="image/png")

st.caption("Simple & cute Text-to-Image app • Hosted on Streamlit Cloud")
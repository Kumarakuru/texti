import streamlit as st
import requests
import io
from PIL import Image
import json

st.set_page_config(page_title="✨ Magic Image Creator", layout="centered")

# --- CONFIGURATION ---
# Ensure this is the EXACT URL from your HF 'Direct Search' or 'CURL' tab
API_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"

def query_endpoint(prompt_text):
    # Standard HF payload for Diffusion models
    payload = {"inputs": prompt_text}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        
        # Check if the response is actually an image
        content_type = response.headers.get("Content-Type", "")
        
        if response.status_code == 200 and "image" in content_type:
            return response.content, "IMAGE"
        else:
            # Return the raw text error so we can read it
            return response.text, "ERROR"
            
    except Exception as e:
        return str(e), "CRASH"

# --- UI ---
st.title("✨ Magic Image Creator")
prompt = st.text_input("Describe your image", value="A futuristic Singapore skyline")

if st.button("Generate"):
    with st.spinner("🎨 Connecting to Endpoint..."):
        data, result_type = query_endpoint(prompt)
        
        if result_type == "IMAGE":
            image = Image.open(io.BytesIO(data))
            st.image(image, caption="Generated Result", use_container_width=True)
            st.success("Success!")
        elif result_type == "ERROR":
            st.error("Server returned an error instead of an image:")
            # This will show you the ACTUAL reason (e.g., 'Model is loading' or 'Out of memory')
            st.code(data, language="json")
            
            if "loading" in data.lower():
                st.info("The GPU is waking up. Please wait 1-2 minutes and try again.")
        else:
            st.error(f"Connection Crash: {data}")

st.divider()
st.caption("Check your HF Dashboard: Status must be 'Running' and Security must be 'Public'.")
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models.segmentation as models
import torchvision.transforms as T
import numpy as np
from PIL import Image
from io import BytesIO
import base64

import os
import requests

model_path = "model/best_model11.pth"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?export=download&id=1R7ad939w_8dBsk5SdJ8ce9Ku7BA3XOpI"
    r = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(r.content)


# ------------------ Streamlit page config ------------------
st.set_page_config(page_title="Vision Extraction", layout="wide")

# ------------------ Custom CSS ------------------
page_bg = """
<style>
    .stApp { 
        background-color: #E0F7FA; 
        text-align: center;
    }
    section[data-testid="stSidebar"] { background-color: #E0F7FA; }
    h1, h2, h3, h4, h5, h6, p, label, span { color: #0D47A1 !important; }
    .caption { color: #0D47A1 !important; text-align: center; margin-top: 5px; }

    /* Columns stack on mobile */
    @media only screen and (max-width: 768px) {
        .stColumns > div {
            width: 100% !important;
            display: block !important;
            text-align: center !important;
        }
    }

    /* Center and enlarge images */
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 12px;
        max-width: 90% !important; /* enlarge a bit */
        height: auto;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------ Title and Explanation ------------------
st.markdown("<h1>Vision Extraction</h1>", unsafe_allow_html=True)
st.markdown("<h2>How Vision Extraction Works</h2>", unsafe_allow_html=True)

# ------------------ Centered static explainer image ------------------
img_path = "Explainer.png"
try:
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_str}">
            <p class="caption">Process: Upload → Preprocess → DeepLabV3 Model → Object Extraction</p>
        </div>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.error(f"Image not found: {img_path}")

# ------------------ File uploader ------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Model ------------------
model = models.deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
state_dict = torch.load("model/best_model11.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# ------------------ Transforms ------------------
val_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# ------------------ Inference function ------------------
def run_inference(img_pil):
    img_resized = img_pil.resize((256, 256))
    img_tensor = val_transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)["out"]
    pred = torch.argmax(output, dim=1)[0].cpu().numpy()

    img_np = np.array(img_resized)
    object_only = np.zeros_like(img_np)
    object_only[pred == 1] = img_np[pred == 1]

    return img_np, object_only

# ------------------ Run inference if file uploaded ------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    original, predicted = run_inference(image)

    # ------------------ Side-by-side layout ------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align:center;'>Original</h3>", unsafe_allow_html=True)
        st.image(original, caption="Original Image", use_column_width=True)

    with col2:
        st.markdown("<h3 style='text-align:center;'>Predicted</h3>", unsafe_allow_html=True)
        st.image(predicted, caption="Predicted Image", use_column_width=True)

    # ------------------ Download button ------------------
    pred_pil = Image.fromarray(predicted)
    buf = BytesIO()
    pred_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Predicted Image",
        data=byte_im,
        file_name="predicted.png",
        mime="image/png"
    )

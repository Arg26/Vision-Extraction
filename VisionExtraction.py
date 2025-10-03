import streamlit as st
import torch
import torch.nn as nn
import torchvision.models.segmentation as models
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from io import BytesIO
import base64
import os
import gdown
import cv2
from streamlit_image_comparison import image_comparison
import requests

# ---------------- CONFIGURATION ----------------
BACKGROUND_IMAGE_PATH = "Background.jpg"
MODEL_PATH = "model/best_model11.pth"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1R7ad939w_8dBsk5SdJ8ce9Ku7BA3XOpI"
MODEL_INPUT_SIZE = (256, 256)
EXAMPLE_IMAGES_DIR = "examples"
EXPLAINER_IMAGE_PATH = "Explainer.png"

# --- CHANGED: App Name ---
st.set_page_config(page_title="SnapSplit", layout="wide")

# ---------------- Helpers ----------------
@st.cache_data
def get_base64_from_local(file_path: str):
    """Encodes a local image file into base64 for CSS embedding."""
    try:
        with open(file_path, "rb") as img_file:
            ext = os.path.splitext(file_path)[1].lower()
            mime_type = "image/png"
            if ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif ext == ".gif":
                mime_type = "image/gif"
            return f"data:{mime_type};base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
    except FileNotFoundError:
        return None


def apply_custom_css():
    base64_bg = get_base64_from_local(BACKGROUND_IMAGE_PATH)
    background_style = (
        f"background-image: url('{base64_bg}');"
        "background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;"
    ) if base64_bg else "background: linear-gradient(to bottom right, #333333, #000000);"

    if not base64_bg:
        st.warning(f"Background image not found at '{BACKGROUND_IMAGE_PATH}'. Using default background.")

    page_bg_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp {{ {background_style} font-family: 'Roboto', sans-serif; }}
        body, .stApp, h1, h2, h3, h4, h5, h6, p, label, span {{ color: #E0E0E0 !important; text-shadow: 1px 1px 3px rgba(0,0,0,0.7); }}
        div[data-testid="stFileUploader"] > div {{ background-color: rgba(40, 40, 40, 0.8); border: 1px dashed #E0E0E0; }}
        .stButton>button, .stDownloadButton>button {{ background-color: rgba(40, 40, 40, 0.8); color: #E0E0E0; border: 1px solid #E0E0E0; transition: all 0.2s ease-in-out; }}
        .stButton>button:hover, .stDownloadButton>button:hover {{ background-color: #E0E0E0; color: #000000; border: 1px solid #E0E0E0; }}
        div[data-testid="stSlider"] div[role="slider"] {{ background-color: #E0E0E0; }}
        div[data-testid="stInfo"] {{ text-align: center; }}
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)


@st.cache_resource
def load_model(device_str: str = "cpu"):
    """Downloads (if needed) and loads the segmentation model into memory."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with st.spinner("Downloading model... This may take a moment."):
            try:
                gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                raise

    device = torch.device(device_str)
    model = models.deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

    try:
        model_state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(model_state, strict=False)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")

    model.to(device)
    model.eval()
    return model


def run_inference(_img_bytes: bytes, model_device: str = "cpu"):
    """Runs the model on the image bytes to get the segmentation mask."""
    image = Image.open(BytesIO(_img_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    device = torch.device(model_device)
    model = load_model(model_device)
    val_transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    original_size = image.size
    img_tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)["out"]
    mask_tensor = torch.argmax(output, dim=1).squeeze(0).cpu().byte()
    mask_upscaler = T.Compose([
        T.ToPILImage(),
        T.Resize(original_size[::-1], interpolation=T.InterpolationMode.NEAREST)
    ])
    mask_pil = mask_upscaler(mask_tensor)
    return image, mask_pil


def apply_post_processing(original_img: Image.Image, mask_pil: Image.Image, effect: str = "None",
                          blur_radius: int = 10, new_bg=None, erosion: int = 0, dilation: int = 0,
                          fixed_bg_color: tuple = None):
    """Applies selected editing effects based on the generated mask."""
    subject_mask_alpha = mask_pil.point(lambda p: 255 if p == 1 else 0).convert("L")
    if erosion > 0 or dilation > 0:
        mask_np = np.array(subject_mask_alpha)
        kernel = np.ones((3, 3), np.uint8)
        if erosion > 0:
            mask_np = cv2.erode(mask_np, kernel, iterations=erosion)
        if dilation > 0:
            mask_np = cv2.dilate(mask_np, kernel, iterations=dilation)
        subject_mask_alpha = Image.fromarray(mask_np.astype(np.uint8))

    if new_bg is not None:
        try:
            if isinstance(new_bg, (bytes, bytearray)):
                new_bg_obj = Image.open(BytesIO(new_bg)).convert("RGB").resize(original_img.size)
            elif hasattr(new_bg, "read"):
                new_bg_obj = Image.open(new_bg).convert("RGB").resize(original_img.size)
            else:
                new_bg_obj = Image.open(new_bg).convert("RGB").resize(original_img.size)
            final_image = new_bg_obj.copy()
            final_image.paste(original_img, (0, 0), subject_mask_alpha)
            return final_image
        except Exception as e:
            st.error(f"Error processing background image: {e}")
            return original_img

    if effect == "Blur":
        blurred_bg = original_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        final_image = blurred_bg.copy()
        final_image.paste(original_img, (0, 0), subject_mask_alpha)
        return final_image
    if effect == "Grayscale":
        grayscale_bg = original_img.convert("L").convert("RGB")
        final_image = grayscale_bg.copy()
        final_image.paste(original_img, (0, 0), subject_mask_alpha)
        return final_image
    if fixed_bg_color:
        background = Image.new('RGB', original_img.size, fixed_bg_color)
        result = background.copy()
        result.paste(original_img, (0, 0), subject_mask_alpha)
        return result
    extracted_subject_np = np.array(original_img)
    mask_np_bool = (np.array(subject_mask_alpha) > 0)
    final_rgba = np.zeros((*extracted_subject_np.shape[:2], 4), dtype=np.uint8)
    final_rgba[mask_np_bool] = np.append(extracted_subject_np[mask_np_bool], np.full((mask_np_bool.sum(), 1), 255), axis=1)
    return Image.fromarray(final_rgba, 'RGBA')

def get_dalle_image(prompt: str = "A scenic landscape"):
    with st.spinner(f"Generating image: '{prompt}'..."):
        image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb"
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            st.session_state.dalle_generated_image_bytes = response.content
            return Image.open(BytesIO(st.session_state.dalle_generated_image_bytes))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download the generated image: {e}")
            return None

# ---------------- App ----------------

def main():
    apply_custom_css()

    def reset_all_states_for_new_image():
        st.session_state.active_feature = None
        st.session_state.effect = "None"
        st.session_state.blur_radius = 10
        st.session_state.erosion = 0
        st.session_state.dilation = 0
        st.session_state.blend_slider = 0.0
        st.session_state.zoom_percentage = 100
        st.session_state.processing_triggered = False
        st.session_state.original_img = None
        st.session_state.mask_pil = None
        st.session_state.processed_image_id = -1
        st.session_state.new_bg_file = None
        st.session_state.dalle_generated_image = None
        st.session_state.dalle_generated_image_bytes = None
        st.session_state.uploaded_bg_file_id = None


    def handle_file_upload():
        uploaded_file = st.session_state.get('file_uploader')
        if uploaded_file is not None:
            new_bytes = uploaded_file.getvalue()
            if st.session_state.last_uploaded_file_id != hash(new_bytes):
                st.session_state.image_bytes = new_bytes
                st.session_state.image_id += 1
                st.session_state.last_uploaded_file_id = hash(new_bytes)
                reset_all_states_for_new_image()
                st.session_state.processing_triggered = True


    # Initialize session state variables
    defaults = {
        "image_bytes": None, "image_id": 0, "active_feature": None,
        "effect": "None", "blur_radius": 10, "erosion": 0, "dilation": 0,
        "blend_slider": 0.0, "new_bg_file": None, "zoom_percentage": 100,
        "dalle_generated_image": None, "original_img": None, "mask_pil": None,
        "processed_image_id": -1, "processing_triggered": False,
        "last_uploaded_file_id": None, "uploaded_bg_file_id": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.title("✨ Try an Example")
        st.markdown("---")
        if os.path.exists(EXAMPLE_IMAGES_DIR):
            for filename in sorted(os.listdir(EXAMPLE_IMAGES_DIR)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(EXAMPLE_IMAGES_DIR, filename)
                    button_label = os.path.splitext(filename)[0].replace('_', ' ').title()
                    st.image(path, caption=f"{button_label} Example", width=250)
                    if st.button(f"Use {button_label}", key=f"example_{filename}"):
                        with open(path, "rb") as f:
                            image_bytes = f.read()
                        if st.session_state.last_uploaded_file_id != hash(image_bytes):
                            st.session_state.image_bytes = image_bytes
                            st.session_state.image_id += 1
                            st.session_state.last_uploaded_file_id = hash(image_bytes)
                            reset_all_states_for_new_image()
                            st.session_state.processing_triggered = True
                            st.rerun()


    # --- CHANGED: App Name ---
    st.markdown("<h1 style='text-align: center;'>SnapSplit</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Isolate the main subject from its background using AI.</h2>", unsafe_allow_html=True)
    explainer_img_data = get_base64_from_local(EXPLAINER_IMAGE_PATH)
    if explainer_img_data:
        st.markdown(
            f'''<div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-bottom: 20px;">
            <img src="{explainer_img_data}" style="max-width: 600px; height: auto; border-radius: 10px;" alt="Explainer Image"></div>''',
            unsafe_allow_html=True
        )
    st.markdown("---")

    st.file_uploader(
        "Upload an image to begin...",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
        key="file_uploader",
        on_change=handle_file_upload
    )

    if st.session_state.image_bytes and not st.session_state.processing_triggered:
        if st.button("🚀 Extract Image", use_container_width=True):
            st.session_state.processing_triggered = True
            st.rerun()

    if st.session_state.image_bytes and st.session_state.processing_triggered:
        if st.session_state.processed_image_id != st.session_state.image_id:
            try:
                with st.spinner("Processing image..."):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    st.session_state.original_img, st.session_state.mask_pil = run_inference(st.session_state.image_bytes, device)
                    st.session_state.processed_image_id = st.session_state.image_id
            except Exception as e:
                st.error(f"Image processing failed: {e}")
                st.session_state.processing_triggered = False
                return

        original_img = st.session_state.original_img
        mask_pil = st.session_state.mask_pil

        if not original_img or not mask_pil:
            st.warning("Still waiting for image to be processed...")
            return

        st.markdown(f"<h3 style='text-align:center;'>Initial Results</h3>", unsafe_allow_html=True)
        _, col1, col2, _ = st.columns([1, 2, 2, 1])
        with col1:
            st.image(original_img, caption="Original Image", width=400)
        with col2:
            st.image(apply_post_processing(original_img, mask_pil, fixed_bg_color=(0,0,0)), caption="Extracted Subject (Black Background)", width=400)
        st.markdown("---")

        st.markdown(f"<h3 style='text-align:center;'>Interactive View</h3>", unsafe_allow_html=True)
        _, view_container, _ = st.columns([1, 5, 1])
        with view_container:
            view_col, adjust_col = st.columns([2, 1])
            with adjust_col:
                st.markdown("#### Adjust View")
                st.slider("Blend Original/Extracted", 0.0, 1.0, key='blend_slider', step=0.05)
                st.slider("Size Scale (%)", 50, 200, key="zoom_percentage", step=5)
            with view_col:
                default_extracted_image = apply_post_processing(original_img, mask_pil)
                scale_factor = st.session_state.zoom_percentage / 100.0
                new_width = max(1, int(original_img.size[0] * scale_factor))
                new_height = max(1, int(original_img.size[1] * scale_factor))
                resampling_filter = Image.Resampling.LANCZOS
                zoomed_original = original_img.resize((new_width, new_height), resample=resampling_filter)
                zoomed_default_extracted = default_extracted_image.resize((new_width, new_height), resample=resampling_filter)

                if zoomed_default_extracted.mode == 'RGBA':
                    extracted_for_blend = Image.new('RGB', zoomed_default_extracted.size, (0, 0, 0))
                    extracted_for_blend.paste(zoomed_default_extracted, (0, 0), zoomed_default_extracted)
                else:
                    extracted_for_blend = zoomed_default_extracted.convert('RGB')
                original_for_blend = zoomed_original.convert('RGB')

                blended_img = Image.blend(original_for_blend, extracted_for_blend, float(st.session_state.blend_slider))

                image_comparison(
                    img1=original_for_blend,
                    img2=blended_img,
                    label1="Original",
                    label2=f"Blended ({int(st.session_state.blend_slider*100)}% Extracted)"
                )

        st.markdown("---")

        st.markdown(f"<h3 style='text-align:center;'>Editing Tools</h3>", unsafe_allow_html=True)
        _, tools_container, _ = st.columns([1, 5, 1])
        with tools_container:
            b1, b2, b3 = st.columns(3)
            if b1.button("🏞️ Replace Background", use_container_width=True): st.session_state.active_feature = "replace_bg"
            if b2.button("✨ Background Effects", use_container_width=True): st.session_state.active_feature = "effects"
            if b3.button("🎛️ Refine Mask", use_container_width=True): st.session_state.active_feature = "refine"

            preview_image = None
            bg_bytes = None

            if st.session_state.active_feature == "replace_bg":
                uploaded_bg_file = st.file_uploader("Upload a new background...", type=["jpg", "png", "jpeg"], key="bg_uploader")

                if uploaded_bg_file is not None:
                    new_bg_bytes = uploaded_bg_file.getvalue()
                    if st.session_state.get('uploaded_bg_file_id') != hash(new_bg_bytes):
                        st.session_state.new_bg_file = uploaded_bg_file
                        st.session_state.uploaded_bg_file_id = hash(new_bg_bytes)
                        st.session_state.dalle_generated_image = None
                        st.session_state.dalle_generated_image_bytes = None
                        st.rerun()
                elif st.session_state.new_bg_file is not None:
                    st.session_state.new_bg_file = None
                    st.session_state.uploaded_bg_file_id = None
                    st.rerun()

                get_image_col, _ = st.columns([1, 3])
                with get_image_col:
                    if st.button("✨ Get Image", key="get_ai_image"):
                        st.session_state.new_bg_file = None
                        st.session_state.uploaded_bg_file_id = None
                        st.session_state.dalle_generated_image = get_dalle_image()
                        st.rerun()

                if st.session_state.new_bg_file:
                    bg_bytes = st.session_state.new_bg_file.getvalue()
                elif 'dalle_generated_image_bytes' in st.session_state and st.session_state.dalle_generated_image_bytes:
                    bg_bytes = st.session_state.dalle_generated_image_bytes

                if bg_bytes:
                    preview_image = apply_post_processing(original_img, mask_pil, new_bg=bg_bytes)
                    st.markdown("#### Replaced Background Preview")
                    st.image(preview_image, caption="Image with Replaced Background", width=600)
                else:
                    st.info("Upload a background image or generate one with AI.")

            elif st.session_state.active_feature == "effects":
                st.selectbox("Select Effect", ["None", "Blur", "Grayscale"], key="effect")
                if st.session_state.effect == "Blur":
                    st.slider("Blur Radius", 0, 20, key="blur_radius")

                preview_image = apply_post_processing(
                    original_img,
                    mask_pil,
                    effect=st.session_state.effect,
                    blur_radius=st.session_state.blur_radius
                )
                st.markdown("#### Effects Preview")
                st.image(preview_image, caption=f"{st.session_state.effect} Effect Applied", width=600)

            elif st.session_state.active_feature == "refine":
                st.slider("Shrink Edges (Erosion)", 0, 10, key="erosion")
                st.slider("Expand Edges (Dilation)", 0, 10, key="dilation")

                preview_image = apply_post_processing(
                    original_img,
                    mask_pil,
                    erosion=st.session_state.erosion,
                    dilation=st.session_state.dilation,
                    fixed_bg_color=(0, 0, 0)
                )
                st.markdown("#### Refined Mask Preview")
                st.image(preview_image, caption="Preview of Refined Edges", width=600)
        st.markdown("---")

        st.markdown("<h3 style='text-align:center;'>Download Options</h3>", unsafe_allow_html=True)
        _, dl_container, _ = st.columns([1, 5, 1])
        with dl_container:
            col_download_current = st.columns(1)[0]
            with col_download_current:
                image_to_download = preview_image if preview_image else apply_post_processing(original_img, mask_pil)

                filename = "adjusted_image.png"
                if st.session_state.active_feature == "replace_bg" and bg_bytes:
                    filename = "replaced_background.png"
                elif st.session_state.active_feature == "effects":
                    filename = "effects_applied.png"
                elif st.session_state.active_feature == "refine":
                    filename = "refined_mask.png"

                buf_final = BytesIO()
                save_format = "PNG"
                if image_to_download.mode == 'RGBA':
                    save_final = image_to_download
                else:
                    save_final = image_to_download.convert("RGB")
                    save_format = "JPEG" if filename.lower().endswith((".jpg", ".jpeg")) else "PNG"

                save_final.save(buf_final, format=save_format)
                st.download_button(
                    "📥 Download Current Edit",
                    buf_final.getvalue(),
                    filename,
                    f"image/{save_format.lower()}",
                    use_container_width=True
                )

            # --- MODIFIED: Added a separate column for clarity ---
            col_transparent, col_black = st.columns(2)
            
            with col_transparent:
                buf_extracted_transparent = BytesIO()
                extracted_transparent = apply_post_processing(original_img, mask_pil)
                extracted_transparent.save(buf_extracted_transparent, format="PNG")
                st.download_button(
                    "📥 Download (Transparent BG)",
                    buf_extracted_transparent.getvalue(),
                    "snapsplit_transparent.png",
                    "image/png",
                    use_container_width=True
                )
            
            # --- ADDED: New download button for black background ---
            with col_black:
                buf_extracted_black = BytesIO()
                extracted_black = apply_post_processing(original_img, mask_pil, fixed_bg_color=(0, 0, 0))
                extracted_black.save(buf_extracted_black, format="PNG")
                st.download_button(
                    "📥 Download (Black BG)",
                    buf_extracted_black.getvalue(),
                    "snapsplit_black_bg.png",
                    "image/png",
                    use_container_width=True
                )
    else:
        st.info("⬆️ Upload an image or use the sidebar to choose an example to begin.")


if __name__ == "__main__":
    main()

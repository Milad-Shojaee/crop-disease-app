import os
import urllib.request

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image


RESNET_MODEL_PATH = "crop_disease_resnet18.pth"
RESNET_MODEL_URL = "https://huggingface.co/Miladdeploy1368/crop-disease-classifier/resolve/main/crop_disease_resnet18.pth"

VIT_MODEL_PATH = "crop_disease_vit_b16.pth"
VIT_MODEL_URL = "https://huggingface.co/Miladdeploy1368/crop-disease-classifier/resolve/main/crop_disease_vit_b16.pth"

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def clean_label(label):
    if "Pepper" in label:
        plant = "Pepper"
    elif "Potato" in label:
        plant = "Potato"
    elif "Tomato" in label:
        plant = "Tomato"
    else:
        plant = "Unknown"

    disease = label
    disease = disease.replace("Pepper__bell___", "")
    disease = disease.replace("Potato___", "")
    disease = disease.replace("Tomato__", "")
    disease = disease.replace("Tomato_", "")
    disease = disease.replace("_", " ")

    return plant, disease


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)


@st.cache_resource
def load_resnet_model():
    download_model(RESNET_MODEL_PATH, RESNET_MODEL_URL)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))

    model.load_state_dict(
        torch.load(
            RESNET_MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model


@st.cache_resource
def load_vit_model():
    download_model(VIT_MODEL_PATH, VIT_MODEL_URL)

    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(
        in_features=768,
        out_features=len(CLASS_NAMES)
    )

    model.load_state_dict(
        torch.load(
            VIT_MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model


def get_topk_predictions(model, image_tensor, k=3):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)

    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        label = CLASS_NAMES[idx.item()]
        plant, disease = clean_label(label)
        confidence = prob.item() * 100

        results.append(
            {
                "label": label,
                "plant": plant,
                "disease": disease,
                "confidence": confidence
            }
        )

    return results


resnet_model = load_resnet_model()
vit_model = load_vit_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🌿 Crop Disease Image Classifier")

st.write(
    "Upload a crop leaf image and compare predictions from two deep learning models: "
    "ResNet-18 and Vision Transformer ViT-B/16."
)

st.info("Supported plants: Tomato, Potato, and Pepper.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    analyze_button = st.button("🔍 Analyze Image")

    if analyze_button:
        with st.spinner("Analyzing image with ResNet-18 and ViT-B/16..."):

            img_tensor = transform(image)
            img_tensor = img_tensor.unsqueeze(0)

            resnet_results = get_topk_predictions(
                resnet_model,
                img_tensor,
                k=3
            )

            vit_results = get_topk_predictions(
                vit_model,
                img_tensor,
                k=3
            )

        resnet_best = resnet_results[0]
        vit_best = vit_results[0]

        st.success("Prediction Complete")

        st.subheader("Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ResNet-18")
            st.write(f"**Predicted Plant:** {resnet_best['plant']}")
            st.write(f"**Predicted Disease:** {resnet_best['disease']}")
            st.write(f"**Confidence:** {resnet_best['confidence']:.2f}%")

        with col2:
            st.markdown("### ViT-B/16")
            st.write(f"**Predicted Plant:** {vit_best['plant']}")
            st.write(f"**Predicted Disease:** {vit_best['disease']}")
            st.write(f"**Confidence:** {vit_best['confidence']:.2f}%")

        if resnet_best["label"] == vit_best["label"]:
            st.success("Both models agree on the same prediction.")
        else:
            st.warning("The two models disagree. Review the Top-3 predictions below.")

        if resnet_best["confidence"] < 80 and vit_best["confidence"] < 80:
            st.warning(
                "Both models have low confidence. This image may belong to an unsupported plant "
                "or disease class."
            )

        st.subheader("ResNet-18 Top-3 Predictions")

        for result in resnet_results:
            st.write(
                f"**Plant:** {result['plant']} | "
                f"**Disease:** {result['disease']} | "
                f"**Confidence:** {result['confidence']:.2f}%"
            )

        st.subheader("ViT-B/16 Top-3 Predictions")

        for result in vit_results:
            st.write(
                f"**Plant:** {result['plant']} | "
                f"**Disease:** {result['disease']} | "
                f"**Confidence:** {result['confidence']:.2f}%"
            )

        if st.button("🔄 Reset / Upload Another Image"):
            st.session_state.uploader_key += 1
            st.rerun()

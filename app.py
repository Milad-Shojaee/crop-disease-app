import os
import urllib.request

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image


MODEL_PATH = "crop_disease_resnet18.pth"
MODEL_URL = "https://huggingface.co/Miladdeploy1368/crop-disease-classifier/resolve/main/crop_disease_resnet18.pth"

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


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASS_NAMES))

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🌿 Crop Disease Image Classifier")

st.write(
    "Upload a crop leaf image and the model will predict the most likely disease."
)

st.info("Supported plants: Tomato, Potato, and Pepper.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)

    best_idx = top_indices[0][0].item()
    best_prob = top_probs[0][0].item() * 100

    best_label = CLASS_NAMES[best_idx]
    plant, disease = clean_label(best_label)

    st.success("Prediction Complete")

    st.subheader("Best Prediction")
    st.write(f"**Predicted Plant:** {plant}")
    st.write(f"**Predicted Disease:** {disease}")
    st.write(f"**Confidence:** {best_prob:.2f}%")

    if best_prob < 80:
        st.warning(
            "The model is not very confident. This image may belong to an unsupported plant "
            "or disease class."
        )

    st.subheader("Top-3 Predictions")

    for prob, idx in zip(top_probs[0], top_indices[0]):
        label = CLASS_NAMES[idx.item()]
        plant_i, disease_i = clean_label(label)
        confidence = prob.item() * 100

        st.write(
            f"**Plant:** {plant_i} | "
            f"**Disease:** {disease_i} | "
            f"**Confidence:** {confidence:.2f}%"
        )
import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import os

# Streamlit config
st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #343a40;
    }
    h1 {
        color: #007bff;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .class-name {
        font-size: 24px;
        font-weight: bold;
        color: #28a745;
    }
    .confidence {
        font-size: 18px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Constants
NUM_CLASSES = 10
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MODEL_PATH = "model.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

model_data = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

st.title("🖼️ Smart Image Classifier")
st.write("Upload an image, and our ResNet-based AI will tell you what it is!")

st.write("### Supported Classes:")
st.write(", ".join([f"**{cls.capitalize()}**" for cls in CLASSES]))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if model_data is None:
        st.error("Model file not found. Please train the model first by running `py train.py`.")
    else:
        st.write("Classifying...")
        model, device = model_data
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        class_idx = predicted.item()
        class_name = CLASSES[class_idx]
        conf_score = confidence.item() * 100
        
        # Results
        st.markdown(f'''
        <div class="prediction-box">
            <div class="class-name">{class_name.capitalize()}</div>
            <div class="confidence">Confidence: {conf_score:.2f}%</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Show top 3 predictions
        st.write("### Top Predictions")
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        for i in range(top3_prob.size(1)):
            st.write(f"- **{CLASSES[top3_catid[0][i].item()].capitalize()}**: {top3_prob[0][i].item()*100:.2f}%")


import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2

from pathlib import Path
from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_model.pth"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# RETINAL PREPROCESSING
# SAME PIPELINE THAT ACHIEVED 10/10
# ============================================================

def crop_black_borders(
    image,
    threshold=10
):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    mask = gray > threshold

    coords = np.argwhere(mask)

    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    return Image.fromarray(
        cropped
    )


def enhance_contrast(image):

    img = np.array(image)

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        [l, a, b]
    )

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return Image.fromarray(
        enhanced
    )


def preprocess_retina(image):

    image = image.convert("RGB")

    image = crop_black_borders(
        image
    )

    image = enhance_contrast(
        image
    )

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    return image


# ============================================================
# NORMALIZATION
# ============================================================

NORMALIZE = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    model = load_model()

    processed = preprocess_retina(
        image
    )

    tensor = NORMALIZE(
        processed
    )

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        confidence = probabilities[
            predicted_class
        ].item()

    return (
        processed,
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "👁️ Diabetic Retinopathy Screening"
)

st.write(
    "AI-assisted diabetic retinopathy "
    "classification from retinal fundus images."
)


# ============================================================
# SYSTEM STATUS
# ============================================================

with st.sidebar:

    st.header("System")

    if torch.cuda.is_available():

        st.success(
            "CUDA GPU Available"
        )

        st.write(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        st.warning(
            "Running on CPU"
        )

    st.write(
        "Model: EfficientNet-B0"
    )

    st.write(
        "Classes: 5"
    )


# ============================================================
# MODEL CHECK
# ============================================================

try:

    model = load_model()

    st.sidebar.success(
        "Model loaded successfully"
    )

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header(
    "Upload Retinal Image"
)

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original Image"
        )

        st.image(
            image,
            
        )

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    try:

        (
            processed,
            predicted_class,
            confidence,
            probabilities
        ) = predict(image)

        with col2:

            st.subheader(
                "Processed Image"
            )

            st.image(
                processed,
                
            )

        # ------------------------------------
        # Result
        # ------------------------------------

        diagnosis = CLASS_NAMES[
            predicted_class
        ]

        st.divider()

        st.header(
            "Screening Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Stage",
                diagnosis
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        # ------------------------------------
        # Probabilities
        # ------------------------------------

        st.subheader(
            "Class Probabilities"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                probabilities[i].item()
                * 100
            )

            st.write(
                f"**{class_name}** — "
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    int(probability),
                    100
                )
            )

        # ------------------------------------
        # Disclaimer
        # ------------------------------------

        st.info(
            "This prototype is intended for "
            "screening/research purposes and "
            "is not a medical diagnosis."
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

else:

    st.info(
        "Upload an APTOS-style retinal "
        "fundus image to begin screening."
    )

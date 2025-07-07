import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Definir transformaciones (mismo val_test_transforms)
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# 3. Cargar modelo entrenado
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 5)  # 5 clases
    )
    state_dict = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# 4. Etiquetas de clases (mismo orden que en entrenamiento)
CLASS_NAMES = [
    'other_activities',
    'safe_driving',
    'talking_phone',
    'texting_phone',
    'turning'
]

# 5. Crear pestañas
tabs = st.tabs(["Clasificación de Conducción", "Predicción de Demanda de Transporte", "Recomendación de Destinos de Viaje"])

# Pestaña 1: Clasificación de imágenes
tab1 = tabs[0]
with tab1:
    st.title("📊 Clasificación de Conducción Distractiva")
    st.write("Sube una imagen de un conductor y el modelo te dirá qué es más probable que se esté haciendo.")

    # 6. Carga de imagen
    uploaded_file = st.file_uploader("Elige una imagen...", type=['png','jpg','jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imagen subida', use_container_width=True)

        # 7. Preprocesar y predecir
        inp = val_transforms(image).unsqueeze(0).to(device)  # [1,3,224,224]
        with torch.no_grad():
            outputs = model(inp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # 8. Mostrar top‑3 predicciones
        top3_idx    = np.argsort(probs)[-3:][::-1]
        top3_probs  = probs[top3_idx]
        top3_labels = [CLASS_NAMES[i] for i in top3_idx]

        st.markdown("### Top‑3 predicciones:")
        for lbl, p in zip(top3_labels, top3_probs):
            st.write(f"- **{lbl}**: {p:.1%}")

        # 9. Barra de probabilidades de todas las clases
        st.bar_chart({cls: float(p) for cls, p in zip(CLASS_NAMES, probs)})

# Pestaña 2: Exploración (vacía)
tab2 = tabs[1]
with tab2:
    st.write("")  # Contenido vacío

# Pestaña 3: Configuración (vacía)
tab3 = tabs[2]
with tab3:
    st.write("")  # Contenido vacío

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import datetime
import joblib
import recommender as rec  # Tu módulo de recomendación

# ---------------------------
# Configuración general
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROUTES = [
    'BABYLON', 'CITY TERMINAL ZONE', 'FAR ROCKAWAY', 'HEMPSTEAD',
    'HUNT. HICKS.', 'LONG BEACH', 'PORT WASHINGTON',
    'RONK. (GREENPT.)', 'WEST HEMPSTEAD'
]
SEQ_LENGTH = 30
HIDDEN_DIM = 50
NUM_LAYERS = 1
branch_ohe_cols = [f'Branch_{r}' for r in ROUTES]
INPUT_SIZE = 1 + len(branch_ohe_cols)

CLASS_NAMES = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']

# ---------------------------
# Definir modelos
# ---------------------------
class MultiRouteRNN(nn.Module):
    def __init__(self, input_size, hidden_dim=50, num_layers=1):
        super(MultiRouteRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(out)
        out = self.fc1(out)
        return out

@st.cache_resource
def load_rnn_model():
    model = MultiRouteRNN(input_size=INPUT_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load('modelo_multiruta_lstm.pt', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_encoders():
    ohe = OneHotEncoder(categories=[ROUTES], sparse_output=False, handle_unknown='ignore')
    ohe.fit(np.array(ROUTES).reshape(-1, 1))
    scaler = joblib.load('scaler.pkl')
    return ohe, scaler

@st.cache_resource
def load_image_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 5)
    )
    state_dict = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_future(model, initial_seq, branch_ohe, scaler, n_steps=30):
    preds = []
    current_seq = initial_seq.copy()
    for _ in range(n_steps):
        seq_x_num = current_seq.reshape(SEQ_LENGTH, 1)
        seq_x_cat = np.repeat(branch_ohe.reshape(1, -1), SEQ_LENGTH, axis=0)
        seq_x = np.concatenate([seq_x_num, seq_x_cat], axis=1)
        input_tensor = torch.tensor(seq_x[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred_norm = model(input_tensor).numpy().flatten()[0]
        pred_inv = scaler.inverse_transform([[pred_norm]])[0, 0]
        preds.append(pred_inv)
        current_seq = np.append(current_seq[1:], pred_norm)
    return preds

# ---------------------------
# Transformaciones imagen
# ---------------------------
val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ---------------------------
# Pestañas
# ---------------------------
tabs = st.tabs([
    "Clasificación de Conducción",
    "Predicción de Demanda de Transporte",
    "Recomendación de Destinos de Viaje"
])

# -------------------------------------
# Pestaña 1: Clasificación de imágenes
# -------------------------------------
with tabs[0]:
    st.title("📊 Clasificación de Conducción Distractiva")
    uploaded_file = st.file_uploader("Elige una imagen...", type=['png','jpg','jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imagen subida', use_container_width=True)
        model_img = load_image_model()
        inp = val_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_img(inp)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_probs = probs[top3_idx]
        top3_labels = [CLASS_NAMES[i] for i in top3_idx]
        st.markdown("### Top‑3 predicciones:")
        for lbl, p in zip(top3_labels, top3_probs):
            st.write(f"- **{lbl}**: {p:.1%}")
        st.bar_chart({cls: float(p) for cls, p in zip(CLASS_NAMES, probs)})

# -------------------------------------
# Pestaña 2: Predicción de demanda
# -------------------------------------
with tabs[1]:
    st.title("Predicción de Demanda por Ruta")
    route = st.selectbox("Seleccione una ruta:", ROUTES)
    date_input = st.date_input(
        "Seleccione una fecha (posterior a 2025-05-01):",
        min_value=datetime.date(2025, 5, 2),
        value=datetime.date(2025, 5, 2)
    )
    if st.button("Predecir demanda"):
        model_rnn = load_rnn_model()
        ohe, scaler = load_encoders()
        branch_ohe_val = ohe.transform([[route]]).flatten()
        initial_seq = np.random.rand(SEQ_LENGTH)  # Reemplaza con datos reales
        preds = predict_future(model_rnn, initial_seq, branch_ohe_val, scaler)
        future_dates = pd.date_range(date_input, periods=30)
        st.subheader(f"Demanda predicha para {route} en {date_input}")
        st.write(f"Demanda predicha: {preds[0]:.2f}")
        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, preds, marker='o', linestyle='--')
        plt.title(f"Predicción de Demanda - Próximos 30 días ({route})")
        plt.xlabel("Fecha")
        plt.ylabel("Total de Pasajeros")
        plt.grid(True)
        st.pyplot(plt)

# -------------------------------------
# Pestaña 3: Recomendación de destinos
# -------------------------------------
with tabs[2]:
    st.title("🗺️ Recomendación de Destinos de Viaje")
    st.write("Obtén recomendaciones personalizadas de destinos.")

    option = st.radio(
        "Selecciona el modo de recomendación:",
        ("Por UserID", "Por Perfil de Usuario")
    )

    if option == "Por UserID":
        user_id_input = st.text_input("Introduce tu UserID:")
        if st.button("Recomendar"):
            if user_id_input:
                user_id_clean = user_id_input.strip()
                try:
                    user_id_int = int(user_id_clean)
                    recommendations = rec.recommend_destinations(user_id_int, top_n=5)
                    if recommendations is not None and not recommendations.empty:
                        st.success("Recomendaciones generadas:")
                        st.table(recommendations)
                    else:
                        st.warning("Usuario no encontrado o sin recomendaciones.")
                except ValueError:
                    st.error("Por favor introduce un número válido como UserID.")
            else:
                st.warning("Por favor introduce un UserID.")
    
    elif option == "Por Perfil de Usuario":
        st.subheader("Completa tu perfil de viaje:")
        preferences_options = ['City', 'Adventure', 'Nature', 'Beaches', 'Historical']
        selected_prefs = st.multiselect("Selecciona tus preferencias:", options=preferences_options)
        gender = st.selectbox("Selecciona tu género:", options=['Male', 'Female'])
        num_adults = st.selectbox("Número de adultos:", options=[1, 2])
        num_children = st.selectbox("Número de niños:", options=[0, 1, 2])

        if st.button("Recomendar por Perfil"):
            if selected_prefs:
                prefs_string = ", ".join(selected_prefs)
                user_profile = {
                    "Preferences": prefs_string,
                    "Gender": gender,
                    "NumberOfAdults": num_adults,
                    "NumberOfChildren": num_children
                }
                recommendations = rec.recommend_by_profile(user_profile, top_n=5)
                if recommendations is not None and not recommendations.empty:
                    st.success("Recomendaciones generadas:")
                    st.table(recommendations)
                else:
                    st.warning("No se pudieron generar recomendaciones.")
            else:
                st.warning("Por favor selecciona al menos una preferencia.")

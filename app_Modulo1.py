import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import datetime
import joblib

# ---------------------------
# Configuración
# ---------------------------
ROUTES = [
    'BABYLON', 'CITY TERMINAL ZONE', 'FAR ROCKAWAY', 'HEMPSTEAD',
    'HUNT. HICKS.', 'LONG BEACH', 'PORT WASHINGTON',
    'RONK. (GREENPT.)', 'WEST HEMPSTEAD'
]
SEQ_LENGTH = 30
HIDDEN_DIM = 50
NUM_LAYERS = 1

# Simula branch_ohe_cols (en la práctica carga el verdadero)
branch_ohe_cols = [f'Branch_{r}' for r in ROUTES]
INPUT_SIZE = 1 + len(branch_ohe_cols)

# ---------------------------
# Definir modelo
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
def load_model():
    model = MultiRouteRNN(input_size=INPUT_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load('modelo_multiruta_lstm.pt', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_encoders():
    # Simula OneHotEncoder y scaler (en la práctica carga los reales)
    ohe = OneHotEncoder(categories=[ROUTES], sparse_output=False, handle_unknown='ignore')

    ohe.fit(np.array(ROUTES).reshape(-1, 1))
    
    # Cargar el scaler real entrenado
    scaler = joblib.load('scaler.pkl')
    return ohe, scaler

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
# Streamlit app
# ---------------------------
st.title("Predicción de Demanda por Ruta")

route = st.selectbox("Seleccione una ruta:", ROUTES)
date_input = st.date_input(
    "Seleccione una fecha (posterior a 2025-05-01):",
    min_value=datetime.date(2025, 5, 2),
    value=datetime.date(2025, 5, 2)
)

if st.button("Predecir demanda"):
    model = load_model()
    ohe, scaler = load_encoders()

    branch_ohe_val = ohe.transform([[route]]).flatten()

    # Simula secuencia inicial (reemplaza con datos reales)
    initial_seq = np.random.rand(SEQ_LENGTH)

    preds = predict_future(model, initial_seq, branch_ohe_val, scaler)

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

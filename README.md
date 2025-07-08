# Trabajo #3: Aplicaciones en sistemas de recomendación e imágenes

# Contenido del repositorio y descruipción breve de los modelos:


 ## Miembros del equipo

- **Wesly Zamira Huertas Salinas** [@WeslyHuertas](https://github.com/WeslyHuertas)
- **Alejandro Torrado Calderón** [@AlejandroTorradoCalderon](https://github.com/AlejandroTorradoCalderon)
- **Juan Pablo Muñoz Jimenez** [@Distorssion](https://github.com/Distorssion)

# Contenido del repositorio y descripción breve de los modelos:
En este repositorio podrá encontrar :

- Un modelo de predicción de demanda basado en series de tiempo que anticipa el uso de rutas durante los próximos 30 días. 
- Un modelo de clasificación de conducción distractiva que identifica comportamientos de riesgo en los conductores a partir de imágenes, como el uso del celular o giros sin precaución. 
- Un modelo de recomendación que sugiere destinos personalizados a los usuarios, basado en su historial de viajes y preferencias previas.
- El código de la página web en la que se implementan estos tres modelos (app_vf.py) cuyo link de la página es: https://sistema-inteligente-integrado-8sbnp9ktadhvqkilqkx2b8.streamlit.app/
  
# Ubicación y descripción de los archivos:
- app_vf.py: Archivo/script principal de la aplicación web.
- README.md: Documento de presentación y guía de uso.
- Cleaned_Travel_Data.csv: Datos procesados de viajes.
- best_model.pth y modelo_multiruta_lstm.pt: Modelos entrenados.
- recommender.py: Módulo de sistema de recomendación.
- requirements.txt: Dependencias del proyecto.
- codes/: Notebooks de cada módulo del proyecto.

```bash
├── app_vf.py
├── README.md
├── __pycache__/
├── .gitignore
├── Cleaned_Travel_Data.csv
├── best_model.pth
├── modelo_multiruta_lstm.pt
├── recommender.py
├── requirements.txt
├── scaler.pkl
└── codes/
    ├── Modulo1.ipynb
    ├── Modulo2.ipynb
    └── Modulo3.ipynb

import gdown
import os

urls = {
    "sift_rf.pkl": "https://drive.google.com/uc?id=1st6UcyTGtyBsjihRXbh6SyFfsJkbNdnm",
    "sift_knn.pkl": "https://drive.google.com/uc?id=1mPIsrRKpI_PwWfk0vgaw_micmveKt0J9",
    "hog_rf.pkl": "https://drive.google.com/uc?id=1IDBYrpk-vFqvO_WlUVqw1jjKUTenlRTF",
    "hog_knn.pkl": "https://drive.google.com/uc?id=1Of0mVADhePDo5GvPn2XYXqC1saIVaWjj",
}

os.makedirs("Modelos_entrenados", exist_ok=True)

for nombre, url in urls.items():
    salida = os.path.join("Modelos_entrenados", nombre)
    if not os.path.exists(salida):
        print(f"Descargando {nombre}...")
        gdown.download(url, salida, quiet=False)
    else:
        print(f"{nombre} ya existe. Omitiendo.")

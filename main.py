import os
import joblib
from Interfaz.Interfaz import ejecutar_app

def cargar_modelos():
    modelos = {}
    ruta_base = 'Modelos_entrenados/'

    archivos = {
        'knn-hog': 'hog_knn.pkl',
        'knn-sift': 'sift_knn.pkl',
        'rf-hog': 'hog_rf.pkl',
        'rf-sift': 'sift_rf.pkl'
    }

    for clave, nombre_archivo in archivos.items():
        try:
            modelos[clave] = joblib.load(os.path.join(ruta_base, nombre_archivo))
        except Exception as e:
            print(f"[!] Error al cargar {nombre_archivo}: {e}")
    return modelos

if __name__ == "__main__":
    modelos = cargar_modelos()
    ejecutar_app(modelos)

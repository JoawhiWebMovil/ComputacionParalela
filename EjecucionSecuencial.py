import os
import time
import cv2 as cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ruta del dataset local
data_path = "Dataset"

# Lectura secuencial de imágenes por clase
def cargarClasesSecuencial(direc):
    imagenesClase = {}
    for clase_nombre in os.listdir(direc):
        clase_path = os.path.join(direc, clase_nombre)
        if os.path.isdir(clase_path):
            imagenes = []
            for nombre_img in os.listdir(clase_path):
                ruta_img = os.path.join(clase_path, nombre_img)
                im = cv.imread(ruta_img)
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
                imagenes.append(im)
            imagenesClase[clase_nombre] = imagenes
    return imagenesClase

# Vectores Característicos
def vectorCaracteristicoSIFT(img):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return np.mean(descriptors, axis=0)

def vectorCaracteristicoHOG(img):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    img_resized = cv.resize(img, (128, 128))
    vector_hog = hog.compute(img_resized)
    return np.ravel(vector_hog)

def vectoresCaracteristicosSecuencial(diccionario, tipo):
    vCaracteristicoClase = {}
    for clase, imagenes in diccionario.items():
        vectores = []
        for img in imagenes:
            if tipo == 'SIFT':
                VC = vectorCaracteristicoSIFT(img)
            elif tipo == 'HOG':
                VC = vectorCaracteristicoHOG(img)
            vectores.append(VC)
        vCaracteristicoClase[clase] = vectores
    return vCaracteristicoClase

# Preparar datos (X, Y)
def prepararDatosParaML(diccionario_vectores):
    X, y = [], []
    for clase, vectores in diccionario_vectores.items():
        for v in vectores:
            X.append(v)
            y.append(clase)
    return np.array(X), np.array(y)

# Entrenamiento y evaluación
def entrenar_y_evaluar(modelo, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

# MAIN
if __name__ == "__main__":
    tiempo_total_inicio = time.time()

    t_inicio_carga = time.time()
    imagenes_dict = cargarClasesSecuencial(data_path)
    t_fin_carga = time.time()

    # SIFT
    t_inicio_vectoresSIFT = time.time()
    vCaracteristicos_dict_SIFT = vectoresCaracteristicosSecuencial(imagenes_dict, 'SIFT')
    t_fin_vectoresSIFT = time.time()

    t_inicio_entrenamientoSIFT = time.time()
    X_sift, y_sift = prepararDatosParaML(vCaracteristicos_dict_SIFT)
    resultados_sift_rf = entrenar_y_evaluar(RandomForestClassifier(n_jobs=1), X_sift, y_sift)
    resultados_sift_knn = entrenar_y_evaluar(KNeighborsClassifier(n_jobs=1), X_sift, y_sift)
    t_fin_entrenamientoSIFT = time.time()

    # HOG
    t_inicio_vectoresHOG = time.time()
    vCaracteristicos_dict_HOG = vectoresCaracteristicosSecuencial(imagenes_dict, 'HOG')
    t_fin_vectoresHOG = time.time()

    t_inicio_entrenamientoHOG = time.time()
    X_hog, y_hog = prepararDatosParaML(vCaracteristicos_dict_HOG)
    resultados_hog_rf = entrenar_y_evaluar(RandomForestClassifier(n_jobs=1), X_hog, y_hog)
    resultados_hog_knn = entrenar_y_evaluar(KNeighborsClassifier(n_jobs=1), X_hog, y_hog)
    t_fin_entrenamientoHOG = time.time()

    tiempo_total_fin = time.time()

    # Mostrar resumen de tiempos
    print(f"- Tiempo de carga del dataset        : {t_fin_carga - t_inicio_carga:.4f} s")

    print("\nRESUMEN DE TIEMPOS SIFT:")
    print(f"- Tiempo de extracción de vectores   : {t_fin_vectoresSIFT - t_inicio_vectoresSIFT:.4f} s")
    print(f"- Tiempo de entrenamiento de modelos : {t_fin_entrenamientoSIFT - t_inicio_entrenamientoSIFT:.4f} s")

    print("\nRESUMEN DE TIEMPOS HOG:")
    print(f"- Tiempo de extracción de vectores   : {t_fin_vectoresHOG - t_inicio_vectoresHOG:.4f} s")
    print(f"- Tiempo de entrenamiento de modelos : {t_fin_entrenamientoHOG - t_inicio_entrenamientoHOG:.4f} s")

    print("\nRESUMEN GENERAL:")
    print(f"- Tiempo TOTAL del proceso           : {tiempo_total_fin - tiempo_total_inicio:.4f} s")

    print("\nRESULTADOS:")
    print(f"{'Modelo':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 68)

    def fila(nombre, res):
        print(f"{nombre:<20} {res['accuracy']*100:10.2f} {res['precision']*100:10.2f} {res['recall']*100:10.2f} {res['f1']*100:10.2f}")

    fila("SIFT_RandomForest", resultados_sift_rf)
    fila("SIFT_KNN", resultados_sift_knn)
    fila("HOG_RandomForest", resultados_hog_rf)
    fila("HOG_KNN", resultados_hog_knn)

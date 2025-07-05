import cv2 as cv
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from Librerias.Extraccion_caracteristicas import ( vectorCaracteristicoHOG, vectorCaracteristicoSIFT )

def extraer_vector(img, extractor):
    if extractor == "sift":
        return vectorCaracteristicoSIFT(img)
    else:
        return vectorCaracteristicoHOG(img)

def leer_imagen(ruta): 
    img = cv.imread(ruta)
    if img is None:
        return None, os.path.basename(ruta)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def clasificar_imagen(ruta, extractor, modelo):
    img = leer_imagen(ruta)
    vector = extraer_vector(img, extractor)
    pred = modelo.predict([vector])[0]
    return pred, os.path.basename(ruta)

def clasificar_imagenes_con_modelo(rutas_imagenes, extractor, modelo):
    resultados = {label: [] for label in modelo.classes_}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futuros = [executor.submit(clasificar_imagen, ruta, extractor, modelo) for ruta in rutas_imagenes]

        for future in futuros:
            pred, nombre_archivo = future.result()
            if pred is not None and pred in resultados:
                resultados[pred].append(nombre_archivo)

    return resultados
import os
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Extracción de vectores característicos
def vectorCaracteristicoSIFT(img):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        return np.mean(descriptors, axis=0)
    else:
        return np.zeros(128)

def vectorCaracteristicoHOG(img):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    img_resized = cv.resize(img, (128, 128))
    vector_hog = hog.compute(img_resized)
    return np.ravel(vector_hog)

def vectorCaracteristicoResultante(clase, tipo):
    VCR = []
    for img in clase:
        if tipo == 'SIFT':
            VC = vectorCaracteristicoSIFT(img)
        elif tipo == 'HOG':
            VC = vectorCaracteristicoHOG(img)
        VCR.append(VC)
    return VCR

def vectoresCaracteristicosParalelizado(diccionario, tipo):
    vCaracteristicoClase = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        features = {
            executor.submit(vectorCaracteristicoResultante, imagenes, tipo): clase
            for clase, imagenes in diccionario.items()
        }
        for feature in features:
            clase = features[feature]
            resultado = feature.result()
            vCaracteristicoClase[clase] = resultado
    return vCaracteristicoClase

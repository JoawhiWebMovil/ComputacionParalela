import os
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor

# Lectura de imágenes
def lecturaDeClase(clase_path):
    imagenes = []
    clase_nombre = os.path.basename(clase_path)
    for d in os.listdir(clase_path):
        im = cv.imread(os.path.join(clase_path, d))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        imagenes.append(im)
    return clase_nombre, imagenes

def cargarClasesParalelizado(direc):
    carpetas = [os.path.join(direc, nombre) for nombre in os.listdir(direc)
                if os.path.isdir(os.path.join(direc, nombre))]
    imagenesClase = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        resultados = executor.map(lecturaDeClase, carpetas)
    for clase, imagenes in resultados:
        imagenesClase[clase] = imagenes
    return imagenesClase
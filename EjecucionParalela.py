import time

from Librerias.Extraccion_caracteristicas import ( vectoresCaracteristicosParalelizado )

from Librerias.Evaluacion import ( prepararDatosParaML, entrenar_y_evaluar_paralelo )

from Librerias.Lectura import ( cargarClasesParalelizado )


# Ruta del dataset local
data_path = "Dataset"

# MAIN 
if __name__ == "__main__":
    tiempo_total_inicio = time.time()

    t_inicio_carga = time.time()
    imagenes_dict = cargarClasesParalelizado(data_path)
    t_fin_carga = time.time()

    # SIFT 
    t_inicio_vectoresSIFT = time.time()
    vCaracteristicos_dict_SIFT = vectoresCaracteristicosParalelizado(imagenes_dict, 'SIFT')
    t_fin_vectoresSIFT = time.time()

    t_inicio_entrenamientoSIFT = time.time()
    X_sift, y_sift = prepararDatosParaML(vCaracteristicos_dict_SIFT)
    resultados_sift_rf, resultados_sift_knn = entrenar_y_evaluar_paralelo(X_sift, y_sift)
    t_fin_entrenamientoSIFT = time.time()

    # HOG 
    t_inicio_vectoresHOG = time.time()
    vCaracteristicos_dict_HOG = vectoresCaracteristicosParalelizado(imagenes_dict, 'HOG')
    t_fin_vectoresHOG = time.time()

    t_inicio_entrenamientoHOG = time.time()
    X_hog, y_hog = prepararDatosParaML(vCaracteristicos_dict_HOG)
    resultados_hog_rf, resultados_hog_knn = entrenar_y_evaluar_paralelo(X_hog, y_hog)
    t_fin_entrenamientoHOG = time.time()

    tiempo_total_fin = time.time()

    #  Mostrar resumen de tiempos 
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
    print("-" * 62)

    def fila(nombre, res):
        print(f"{nombre:<20} {res['accuracy']*100:10.2f} {res['precision']*100:10.2f} {res['recall']*100:10.2f} {res['f1']*100:10.2f}")

    fila("SIFT_RandomForest", resultados_sift_rf)
    fila("SIFT_KNN", resultados_sift_knn)
    fila("HOG_RandomForest", resultados_hog_rf)
    fila("HOG_KNN", resultados_hog_knn)

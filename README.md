# Clasificación de Imágenes con SIFT/HOG y KNN/Random Forest

Este proyecto permite clasificar imágenes en dos categorías (`Bike` y `Car`) utilizando técnicas de extracción de características (`SIFT`, `HOG`) combinadas con modelos de clasificación (`KNN`, `RandomForest`).
Se han implementado versiones de entrenamiento **paralelo** y **secuencial**.
Además de una interfaz gráfica para clasificar imágenes de manera paralela.

---

## 📁 Estructura del Proyecto

```
ComputacionParalela/
├── Dataset/                     # Contiene las carpetas Bike/ y Car/
├── Entrenamiento/               # Contiene archivos donde se puede entrenar modelos de manera secuencial y paralela
│   ├── EntrenamientoParalela.py
│   └── EntrenamientoSecuencial.py
├── Interfaz/
│   ├── Interfaz.py
│   └── Clasificador.py
├── Librerias/
│   ├── __init__.py
│   ├── Extraccion_caracteristicas.py
│   ├── Evaluacion.py
│   └── Lectura.py
├── Modelos_entrenados/         # Descargar esta carpeta ejecutando "descargar_modelos.py"
├── img/                        # Contiene bike.jpeg y car.jpeg para la interfaz
├── main.py                     # Archivo principal a ejecutar para mostrar interfaz
└── README.md
```

---

### 🔽 Descarga de modelos

Para descargar los modelos necesarios:

```bash
pip install gdown
python descargar_modelos.py
```

## 🚀Ejecución del Proyecto

### 1. Ejecutar la interfaz gráfica

```
python main.py
```
Permite:
- Agregar imágenes individuales.
- Elegir un extractor (SIFT o HOG).
- Elegir un modelo (KNN o Random Forest).
- Clasificación en paralelo de gran cantidad de imágenes.
- Visualizar la clasificación en una ventana dividida.

### 2. Entrenamiento Paralelo
``` 
python Entrenamiento/EntrenamientoParalela.py
```
Este script:
- Carga el dataset usando múltiples hilos.
- Extrae características (SIFT y HOG) en paralelo.
- Entrena KNN y RandomForest simultáneamente.
- Muestra métricas de desempeño y tiempos de ejecución.
- El entrenamiento paralelo usa `ThreadPoolExecutor` para acelerar carga, extracción y entrenamiento.

### 3. Entrenamiento Secuencial
```
python Entrenamiento/EntrenamientoSecuencial.py
```
Este script ejecuta el mismo flujo, pero usando procesamiento secuencial para comparar los tiempos.

---

## 🧠 Modelos Entrenados

Se generan y guardan 4 modelos:

| Modelo         | Descripción                      | Archivo Pkl               |
|----------------|----------------------------------|---------------------------|
| SIFT + KNN     | SIFT + K Nearest Neighbors       | `sift_knn.pkl`            |
| SIFT + RF      | SIFT + Random Forest             | `sift_rf.pkl`             |
| HOG + KNN      | HOG + K Nearest Neighbors        | `hog_knn.pkl`             |
| HOG + RF       | HOG + Random Forest              | `hog_rf.pkl`              |

Estos modelos se cargan automáticamente en `main.py` si están presentes en la carpeta `Modelos_entrenados/`.

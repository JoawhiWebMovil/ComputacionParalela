from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Prepara los datos (X, y) a partir de un diccionario de vectores
def prepararDatosParaML(diccionario_vectores):
    X, y = [], []
    for clase, vectores in diccionario_vectores.items():
        for v in vectores:
            X.append(v)
            y.append(clase)
    return np.array(X), np.array(y)

# Entrena y evalúa un modelo individual
def entrenar_y_evaluar(modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    return {
        'modelo': type(modelo).__name__,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

# Ejecuta entrenamiento y evaluación en paralelo para varios modelos
def entrenar_y_evaluar_paralelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelos = [
        RandomForestClassifier(),
        KNeighborsClassifier()
    ]

    resultados = []
    with ThreadPoolExecutor() as executor:
        futuros = [executor.submit(entrenar_y_evaluar, modelo, X_train, X_test, y_train, y_test) for modelo in modelos]
        for futuro in futuros:
            resultados.append(futuro.result())

    return resultados[0], resultados[1]

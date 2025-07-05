import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk

from .Clasificador import ( clasificar_imagenes_con_modelo )


class InterfazClasificacion:
    def __init__(self, root, modelos):
        self.root = root
        self.root.title("Clasificación de imágenes")
        self.root.geometry("800x550")
        self.root.config(padx=20, pady=20)
        self.modelos = modelos

        # Variables
        self.rutas_imagenes = []

        self.var_sift = tk.BooleanVar()
        self.var_hog = tk.BooleanVar()
        self.var_rf = tk.BooleanVar()
        self.var_knn = tk.BooleanVar()

        self._crear_widgets()
        self._configurar_dependencias()

    def _crear_widgets(self):
        titulo = tk.Label(self.root, text="Clasificación de imágenes", font=("Helvetica", 16))
        titulo.pack(pady=10)

        contenedor = tk.Frame(self.root)
        contenedor.pack(pady=10)

        # Panel izquierdo - Lista de imágenes
        frame_izq = tk.Frame(contenedor)
        frame_izq.grid(row=0, column=0, padx=30)

        btn_agregar = tk.Button(frame_izq, text="Agregar imagen(es)", command=self.agregar_imagenes, width=20, height=2)
        btn_agregar.pack(pady=10)

        self.lista_imagenes = tk.Listbox(frame_izq, height=12, width=40)
        self.lista_imagenes.pack()

        # Panel central - Extractores
        frame_extractores = tk.Frame(contenedor)
        frame_extractores.grid(row=0, column=1, padx=30)

        tk.Label(frame_extractores, text="Extractor de característica", font=("Helvetica", 12)).pack(pady=5)
        tk.Checkbutton(frame_extractores, text="SIFT", variable=self.var_sift).pack(anchor="w")
        tk.Checkbutton(frame_extractores, text="HOG", variable=self.var_hog).pack(anchor="w")

        # Panel derecho - Modelos
        frame_modelos = tk.Frame(contenedor)
        frame_modelos.grid(row=0, column=2, padx=30)

        tk.Label(frame_modelos, text="Modelos", font=("Helvetica", 12)).pack(pady=5)
        tk.Checkbutton(frame_modelos, text="RandomForest", variable=self.var_rf).pack(anchor="w")
        tk.Checkbutton(frame_modelos, text="Knn", variable=self.var_knn).pack(anchor="w")

        # Botón Clasificar
        btn_clasificar = tk.Button(self.root, text="Clasificar", command=self.clasificar_imagenes, width=15, height=2)
        btn_clasificar.pack(pady=20)

    def _configurar_dependencias(self):
        self.var_sift.trace_add('write', self._desmarcar_hog_si_sift)
        self.var_hog.trace_add('write', self._desmarcar_sift_si_hog)
        self.var_knn.trace_add('write', self._desmarcar_rf_si_knn)
        self.var_rf.trace_add('write', self._desmarcar_knn_si_rf)

    def _desmarcar_hog_si_sift(self, *args):
        if self.var_sift.get():
            self.var_hog.set(False)

    def _desmarcar_sift_si_hog(self, *args):
        if self.var_hog.get():
            self.var_sift.set(False)

    def _desmarcar_rf_si_knn(self, *args):
        if self.var_knn.get():
            self.var_rf.set(False)

    def _desmarcar_knn_si_rf(self, *args):
        if self.var_rf.get():
            self.var_knn.set(False)

    def agregar_imagenes(self):
        rutas = filedialog.askopenfilenames(
            title="Seleccionar imágenes",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if rutas:
            for ruta in rutas:
                self.rutas_imagenes.append(ruta)
                nombre_archivo = os.path.basename(ruta)
                self.lista_imagenes.insert(tk.END, f"*{nombre_archivo}")

    def clasificar_imagenes(self):
        extractores = []
        if self.var_sift.get(): extractores.append('sift')
        if self.var_hog.get(): extractores.append('hog')
        clasificadores = []
        if self.var_knn.get(): clasificadores.append('knn')
        if self.var_rf.get(): clasificadores.append('rf')

        if len(extractores) != 1 or len(clasificadores) != 1:
            messagebox.showwarning("Selección inválida", "Debe seleccionar exactamente un extractor (SIFT o HOG) y un modelo (KNN o RF).")
            return

        extractor = extractores[0]
        clasificador = clasificadores[0]
        clave_modelo = f"{clasificador}-{extractor}"
        modelo = self.modelos.get(clave_modelo)

        if modelo is None:
            messagebox.showerror("Error", f"No se encontró el modelo '{clave_modelo}'")
            return

        resultados = clasificar_imagenes_con_modelo(self.rutas_imagenes, extractor, modelo)
        self.mostrar_resultados(clave_modelo.upper(), resultados)

    def mostrar_resultados(self, titulo, resultados):
        nueva_ventana = tk.Toplevel(self.root)
        nueva_ventana.title(titulo)
        nueva_ventana.geometry("500x400")

        contenedor = tk.Frame(nueva_ventana)
        contenedor.pack(fill="both", expand=True)

        frame_izq = tk.Frame(contenedor, padx=20, pady=10)
        frame_izq.grid(row=0, column=0, sticky="n")

        # Línea divisoria
        frame_linea = tk.Frame(contenedor, width=2, bg="gray")
        frame_linea.grid(row=0, column=1, sticky="ns", padx=5)

        frame_der = tk.Frame(contenedor, padx=20, pady=10)
        frame_der.grid(row=0, column=2, sticky="n")

        # Imágenes de referencia (más grandes)
        img_bike = Image.open("img/bike.jpeg").resize((150, 150))
        img_bike = ImageTk.PhotoImage(img_bike)
        tk.Label(frame_izq, image=img_bike).pack()
        frame_izq.image = img_bike

        img_car = Image.open("img/car.jpeg").resize((150, 150))
        img_car = ImageTk.PhotoImage(img_car)
        tk.Label(frame_der, image=img_car).pack()
        frame_der.image = img_car

        # Texto (color negro)
        for nombre in resultados.get("Bike", []) + resultados.get("Bike", []):
            tk.Label(frame_izq, text=nombre, fg="black").pack(anchor="w")

        for nombre in resultados.get("Car", []) + resultados.get("Car", []):
            tk.Label(frame_der, text=nombre, fg="black").pack(anchor="w")



def ejecutar_app(modelos):
    root = tk.Tk()
    app = InterfazClasificacion(root, modelos)
    root.mainloop()

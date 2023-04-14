# base de datos para ver en una camara web en vivo
import os

from cv2 import cv2
import face_recognition as fr

# crear base de datos de imagenes
ruta = 'Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

for nombre in lista_empleados:
    # Lee las imagenes segun el directorio
    imagen_actual = cv2.imread(f'{ruta}\{nombre}')
    mis_imagenes.append(imagen_actual)
    # separa el texto de la extencion del archivo
    nombres_empleados.append(os.path.splitext(nombre)[0])


# Codificar Imagenes
def codificar(imagenes):
    # Crear una lista nueva de
    lista_codificada = []

    # pasar todas las imagenes a rgb
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # codificar el rostro que exista en la imagen
        codificado = fr.face_encodings(imagen)[0]
        # Agregar a la lista de iamgens codificadas
        lista_codificada.append(codificado)

    return lista_codificada


lista_empleados_codificada = codificar(mis_imagenes)


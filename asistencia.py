# base de datos para ver en una camara web en vivo
import os

import numpy
from cv2 import cv2
import face_recognition as fr

from datetime import datetime

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

# tomar ina imagen de camara web
# 0 id
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Leer la imagen de la camara


def registrar_ingresos(persona):
    f = open('registro.csv', 'r+')
    lista_datos = f.readlines()
    nombres_registro = []
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])

    if persona not in nombres_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M:%S')
        f.writelines(f'\n{persona}, {string_ahora}')
exito, imagen = captura.read()
if not exito:
    print('No se a podifo tomar la captura')
else:
    # Reconocer cara en captura
    cara_captura = fr.face_locations(imagen)

    # codificar la cara que halla capturado
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)

    # Buscar coincidencia del rosto con neustra base de datos, zip easy
    for caracodificada, caraubicada in zip(cara_captura_codificada, cara_captura):
        # comrparamos caras codigficadas de nuestra base de datos y cara codificada
        coincidencias = fr.compare_faces(lista_empleados_codificada, caracodificada)
        distancias = fr.face_distance(lista_empleados_codificada, caracodificada)
        # toma el que tien menor valor

        indice_coincidencia = numpy.argmin(distancias)  # retorna un decimal
        print(indice_coincidencia)
        # Mostrar coincidencias si las hay

        if distancias[indice_coincidencia] > 0.6:
            print(" No tienen ninguna coincidencia ")
        else:
            # buscar el nombre del empleado encontrado
            nombre = nombres_empleados[indice_coincidencia]

            # mostrar los puntos de las posiciones
            y1, x2, y2, x1 = caraubicada
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imagen, (x1, y2 - 5), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1 - 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255, 255), 2)

            registrar_ingresos(nombre)
            # mostrar la imagen obtrenida
            cv2.imshow('Imagen Web', imagen)
            # mantener la ventana abierta
            cv2.waitKey(0)


# registrar los ingresos

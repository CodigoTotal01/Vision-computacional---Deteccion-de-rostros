import face_recognition as fr
from cv2 import cv2

# Cargar Imagenes
foto_control = fr.load_image_file('Empleados\\normal.jpg')
foto_prueba = fr.load_image_file('Empleados\\feacarita.jpg')

# El color de la imagen debe estar en rgb
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# Localizar cara control A
lugar_cara_A = fr.face_locations(foto_control)[0]
cara_codficada_A = fr.face_encodings(foto_control)[0]
# marcar en la foto donde esta la cara
cv2.rectangle(foto_control,
              (lugar_cara_A[3], lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0, 255, 0),  # color
              2)  # grosor

# Localizar cara prueba A
lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codficada_B = fr.face_encodings(foto_prueba)[0]
# marcar en la foto donde esta la cara
cv2.rectangle(foto_prueba,
              (lugar_cara_B[3], lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0, 255, 0),  # color
              2)  # grosor

# relizar comparacion - tolerancia
resultado = fr.compare_faces([cara_codficada_A], cara_codficada_B)
# si es menor a 0.6 entonces vale verga - 0.53339318] disntancia
print(resultado)

# Medida de la distanacia
distancia = fr.face_distance([cara_codficada_A], cara_codficada_B)

# mostrar la distancia en la imagen
cv2.putText(foto_prueba,
            f'{resultado} {distancia.round(2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2
            )

# Mostrar imagens
cv2.imshow('Foto Cotnrol', foto_control)
cv2.imshow('Foto prueba', foto_prueba)

print(distancia)

# Mantener el programa abierto
cv2.waitKey()

# Segmentación de Objetos con OpenCV

Trabajo de Visión por Computador realizado por David Cornejo. Desarrollado con Python en Google Colab, haciendo uso de las librerías "cv2" y "numpy" y utilizando el dataset de imágenes "COCO".

---

## Mask R-CNN:
Mask R-CNN es una red neuronal de imágenes fácil de entrenar, que realiza la clasificación de varios objetos de una imagen y devuelve como resultado las coordenadas de dichos objetos ubicados en la misma, así como sus máscaras. 

Según la documentación de Mask R-CNN, contiene varios datasets de imágenes que podemos utilizar, uno de ellos es “COCO” (Microsoft Common Objects in Context), un dataset de detección y segmentación de objetos a gran escala que contiene 328 mil imágenes.
Éste dataset puede realizar detección de bordes de objetos, con la capacidad de clacificarlos mediante 80 categorías de objetos; detección de puntos-clave, conteniendo más de 200 mil imágenes y 250 mil instancias de personas, clasificados en 17 posibles puntos-clave (ojo izquierdo, ojo derecho, nariz, etc), bastante útiles para la clasificación de gestos o emociones; “Panoptic”, capaz de realizar segmentación de objetos, contando con 80 categorías de objetos (personas, animales, cosas) y 91 subcategorías (césped, cielo, calles).
Para poder hacer uso de éste dataset en OpenCV y realizar una segmentación de objetos en una imagen cualquiera, realizaremos el siguiente procedimiento:

1.	Importamos el paquete de librerías de OpenCV y numpy, y cargamos los módulos que contienen el dataset “COCO”. Los módulos necesarios son:

-	frozen_inference_graph_coco.pb
-	mask_rcnn_inception_v2_coco_2018_01_28.pbtxt

```py
import cv2
import numpy as np
```

```py
# Cargamos los modulos de Mask DNN que contienen el modelo neuronal entrenado
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph_coco.pb",
									                  "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
```

2.	Cargamos la imagen en donde realizaremos la segmentación de objetos. Almacenamos su largo y ancho en variables:

```py
# Cargamos la imagen en donde queremos realizar la segmentación
img = cv2.imread("horse.jpg")
height, width, _ = img.shape
```

3.	Antes de nada, creamos una imagen totalmente negra, con las mismas dimensiones de la imagen ingresada. Ésta imagen negra nos servirá para la creación de las máscaras.

```py
# Creamos una imágen totalmente negra, que nos será util para la generación de máscaras
black_image = np.zeros((height, width, 3), np.uint8)
black_image[:] = (100, 100, 0)
```

4.	Para que la imagen sea legible para la red neuronal, debemos convertirla a un formato especial (blob), con esto evitaremos errores de lectura de imagen. Dentro del paquete de librerías OpenCV contenemos ya un método que nos ayuda con el cambio de formato de imagen a blob:

```py
# Convertimos la imagen a un formato que sea legible para el modelo
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
```

5.	Al enviar nuestra imagen al modelo neuronal, éste nos devolverá como resultado una serie de coordenadas que pertenecen a los objetos que han sido detectados por el modelo. Para poder acceder a todas las coordenadas, tenemos que valernos de un bucle for:

```py
# La red neuronal devolverá como resultado un conjunto de objetos detectados en la imagen, por lo que hay que utilizar un bucle para poder recorrerlos
for i in range(detection_count):
	box = boxes[0, 0, i]
	class_id = box[1]
	score = box[2]
	if score < 0.5:
		continue

	# Coordenadas del objeto detectado dentro de la imagen
	x = int(box[3] * width)
	y = int(box[4] * height)
	x2 = int(box[5] * width)
	y2 = int(box[6] * height)

	roi = black_image[y: y2, x: x2]
	roi_height, roi_width, _ = roi.shape
```

Si utilizamos éstas coordenadas en conjunto con un método que nos permita generar rectángulos (Rectangle, de OpenCV) ya podemos mostrar el resultado de la detección de objetos en la imagen, simplemente enviamos las coordenadas devueltas por el modelo, al método de dibujado de rectángulos, y obtenemos lo siguiente:






	






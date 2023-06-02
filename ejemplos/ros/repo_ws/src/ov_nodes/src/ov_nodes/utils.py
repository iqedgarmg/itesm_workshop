#Librerias basicas
import numpy as np

#Herramientas de inferencia de OpenVINO
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore

#Procesamiento de imagenes
import cv2
from PIL import Image as PImage

#Redimensionar imagen manteniendo aspecto original
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), PImage.BICUBIC)
    new_image = PImage.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

#Preprocesamiento general
def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data
    
#Filtrar salida del modelo
def yolo_post(scores, boxes, indices, thresh):

  #Inicializar salidas
  out_boxes, out_scores, out_classes = [], [], []

  #Extraer salidas
  for idx_ in indices:

      #Filtar de acuerdo a threshold
      if(scores[tuple(idx_)] > thresh):

        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])

  return out_boxes, out_scores, out_classes

#Dibujar predicciones
def draw_boxes(img, out_boxes, labels=None, colors=None, width=10):

  #Dibujar rectangulos de colores
  for i in range(len(out_boxes)): 
    img = cv2.rectangle(img, (int(out_boxes[i][1]), int(out_boxes[i][0])), (int(out_boxes[i][3]), int(out_boxes[i][2])), colors[i], width)

  return img
  
#Cargar modelo IR 
def load_IR_to_IE(model_xml, model_bin):

  #Inicialiar Inference Engine
  plugin = IECore()

  #Cargar modelo
  net = plugin.read_network(model=model_xml, weights=model_bin)
  exec_net = plugin.load_network(network=net, device_name="CPU", num_requests=2)
  print("Modelo cargado a IE")
  return exec_net

#Inferencia sincrona
def synchronous_inference(executable_net, image, shape):

  #Correr inferencia
  result = executable_net.infer(inputs = {"input_1": image, 'image_shape': shape})

  return result

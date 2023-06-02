"""
YOLO OpenVINO implementation

Authors: 
  * Edgar Macias Garcia (edgar.macias.garcia@intel.com)
"""

#!/usr/bin/env python

#Import ROS dependencies
import rospy

#Import utils
import cv2
import numpy as np
from PIL import Image as PImage
import time

#Import messages
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ov_nodes.utils import *

#Herramientas de inferencia de OpenVINO
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore


#Image publisher
def pub_img(img):

  global seg_pub, cvi
  
  #Convert image
  cvi.image = img
  im_msg = cvi.cv2_to_imgmsg(cvi.image, encoding="bgr8")
  
  #Construir mensaje
  im_msg.header.stamp = rospy.get_rostime()
  im_msg.encoding = "bgr8";

  #Publicar mensaje
  seg_pub.publish(im_msg)

#Image callback
def ImageCallback(msg):

  global cvi, model

  #ROS Image to CV
  img = cvi.imgmsg_to_cv2(msg, "bgr8")
  
  #CV to PIL
  #image = PImage.fromarray(img)  
  image = PImage.fromarray(np.uint8(img))
  
  #Preprocesar entrada
  image_data = preprocess(image)
  image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
  
  #Someter modelo a entrada
  start = time.time()
  output = synchronous_inference(model, image_data, image_size)
  end = time.time()
  print("Tiempo de inferencia: " + str(end - start) + " s")

  #Obtener salidas por tipo
  boxes = output["yolonms_layer_1"]
  scores = output["yolonms_layer_1:1"]
  indices = output["yolonms_layer_1:2"][0]
  
  #Obtener predicciones que cumplen threshold
  out_boxes, out_scores, out_classes = yolo_post(scores, boxes, indices, 0.01)

  #Convertir imagen RGB a Numpy
  image = np.array(image)
  
  labels = ["person"]
  colors = [(255, 0, 0)]

  #Dibujar rectangulos
  img_int = draw_boxes(image, out_boxes, labels, colors, 10)
  
  #Publicar segmentacion
  pub_img(img_int)

#Main
def main():

  #Globals
  global seg_pub, model, cvi

  #init node
  rospy.init_node('image_suscriber_py')
  
  #Init cv bridge
  cvi = CvBridge()
  
  #Init image subscriber
  rospy.Subscriber("/camera/rgb", Image, ImageCallback)
  
  #Init publishers
  seg_pub = rospy.Publisher("/yolo_ov/img_seg", Image, queue_size = 1)
  
  #Cargar modelo usando IE
  path = "/home/edgarmg/Projects/itesm_workshop/ejemplos/ros/repo_ws/src/ov_nodes/"
  model = load_IR_to_IE(path + "models/tiny-yolov3-11.xml", path + "models/tiny-yolov3-11.bin")

  #Set node rate
  rate = rospy.Rate(30)

  #Main loop
  while not rospy.is_shutdown():

    #Run callbacks
    rospy.spin()

if __name__ == '__main__':
  main()

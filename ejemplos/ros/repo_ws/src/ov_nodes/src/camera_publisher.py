#Importar rospy
import rospy 

#Importar utilidades
import numpy as np

#Utilidades de Vision
import cv2

#Utilidades de ROS
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def main():

  #Inicializar nodo
  rospy.init_node("camera_publisher")

  #Crear instancia de clase Publisher
  emisor = rospy.Publisher("/camera/rgb", Image, queue_size=1)
  
  #Crear objeto cvbridge
  cvi = CvBridge()

  #Acceder a camara con ID 0
  input_video = cv2.VideoCapture(0)

  #Inicializar indice de publicacion (timer)
  rate = rospy.Rate(15)


  #Crear ciclo
  while not rospy.is_shutdown():
  
    #Obtener imagen de la camara
    ret_val, cvi.image = input_video.read()

    #Crear mensaje
    im_msg = cvi.cv2_to_imgmsg(cvi.image, encoding="bgr8")

    #Construir mensaje
    im_msg.header.stamp = rospy.get_rostime()
    im_msg.encoding = "bgr8";

    #Publicar mensaje
    emisor.publish(im_msg)
    
    #Retardo
    rate.sleep()
    
if __name__ == '__main__':
  main()

import cv2
import numpy as np
from tensorflow.keras.models import  load_model
import tensorflow as tf
tf.compat.v1.reset_default_graph()
#from keras.optimizers import SGD

def pneumoniatest():
 img_size=50
 filepath='D:/NAVTTCH/Pneumonia Detection using chest xrays images/Pneumonia_Detection_savedmodel_with_97p'
 model=load_model(filepath, compile=True)

 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 print("model loaded :)")
 
 testimagepath="D:/FYP with Updated DataSet/Split_Covid-Pneumonia_Dataset/test/Viral Pneumonia/person1667_virus_2881.jpeg"
 img = cv2.imread(testimagepath)
 img = cv2.resize(img,(img_size,img_size))
 img = np.reshape(img,[1,img_size,img_size,3])
 img = np.array(img)
 classes = model.predict_classes(img)
 return(classes)
 
 
a=pneumoniatest()
if a==0:
    print("Normal")
elif a==1:
     print("Pnemonia")
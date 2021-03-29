# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:17:22 2020

@author: nabeel
"""



import splitfolders
path = "D:/NAVTTCH/Pneumonia Detection using chest xrays images/Dataset_pneuminia_normal"


splitfolders.ratio(path, output="train_test_Pneumonia_Dataset", seed=1337,ratio=(.7, .1,.2))
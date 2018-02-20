import matplotlib
import matplotlib.pyplot as pylab
import csv
import os
import numpy as np


cnn_layers = ['conv1','conv2','conv3','conv4','fulcon1']


adacnn_sizes = [np.array([38,30,36,64,56],dtype=np.float32),
                np.array([52,24,54,40,58],dtype=np.float32),
                np.array([102,74,94,128,108],dtype=np.float32),
                np.array([98,64,82,120,88],dtype=np.float32),
                np.array(300,dtype=np.float32)]

fixed_sizes = [64,64,128,128,300]
filter_sizes = []

adacnn_mean_sizes = [np.mean(a) for a in adacnn_sizes]
adacnn_std_sizes = [np.std(a) for a in adacnn_sizes]




import matplotlib
import matplotlib.pyplot as pylab
import csv
import os
import numpy as np


cnn_layers = ['conv1','conv2','conv3','conv4','fulcon1']


adacnn_sizes = [np.array(3,dtype=np.float32),np.array([38,30,36,64,56],dtype=np.float32),
                np.array([52,24,54,40,58],dtype=np.float32),
                np.array([102,74,94,128,108],dtype=np.float32),
                np.array([98,64,82,120,88],dtype=np.float32),
                np.array(300,dtype=np.float32)]

fixed_sizes = [3,64,64,128,128,300]
filter_sizes = [2*4,4*8,2*4,2*2,16*8]

adacnn_mean_sizes = [np.mean(a) for a in adacnn_sizes]
adacnn_std_sizes = [np.std(a) for a in adacnn_sizes]

adacnn_mean_parameters = []
adacnn_std_parameters = []
fixed_parameters = []

adacnn_mean_total = 0
adacnn_std_total = 0
fixed_total = 0
for l_i in range(1,len(adacnn_sizes)-1):
    print(l_i)
    if 'conv' in cnn_layers[l_i-1]:
        adacnn_mean_parameters.append(filter_sizes[l_i-1]*adacnn_mean_sizes[l_i-1]*adacnn_mean_sizes[l_i])
        adacnn_std_parameters.append(filter_sizes[l_i-1]*adacnn_std_sizes[l_i-1]*adacnn_std_sizes[l_i])
        fixed_parameters.append(filter_sizes[l_i-1]*fixed_sizes[l_i-1]*fixed_sizes[l_i])

        adacnn_mean_total += filter_sizes[l_i-1]*adacnn_mean_sizes[l_i-1]*adacnn_mean_sizes[l_i]
        adacnn_std_total += filter_sizes[l_i-1]*adacnn_std_sizes[l_i-1]*adacnn_std_sizes[l_i]
        fixed_total += filter_sizes[l_i-1]*fixed_sizes[l_i-1]*fixed_sizes[l_i]
    elif 'fulcon'in cnn_layers[l_i-1]:
        #adacnn_mean_parameters.append(filter_sizes[l_i-1] * adacnn_mean_sizes[l_i - 1] * adacnn_mean_sizes[l_i])
        #adacnn_std_parameters.append(filter_sizes[l_i-1] * adacnn_std_sizes[l_i - 1] * adacnn_std_sizes[l_i])
        #fixed_parameters.append(filter_sizes[l_i-1] * fixed_sizes[l_i - 1] * fixed_sizes[l_i])
        adacnn_mean_total += filter_sizes[l_i-1] * adacnn_mean_sizes[l_i - 1] * adacnn_mean_sizes[l_i]
        adacnn_std_total += filter_sizes[l_i-1] * adacnn_std_sizes[l_i - 1] * adacnn_std_sizes[l_i]
        fixed_total += filter_sizes[l_i-1] * fixed_sizes[l_i - 1] * fixed_sizes[l_i]

f,ax = pylab.subplots(1,2)

print(adacnn_mean_parameters)
print(fixed_parameters)
width=0.35
ax[0].bar(np.arange(len(cnn_layers)-1),np.array(adacnn_mean_parameters),width,yerr=adacnn_std_parameters,label='AdaCNN')
ax[0].bar(np.arange(len(cnn_layers)-1)+width,np.array(fixed_parameters),width,label='Fixed-CNN')
ax[0].set_title('Parameters of Different\nConvolution Layers')
#ax[0].set_xticks(np.arange(len(cnn_layers)-1) + width / 2, ('conv1', 'conv2', 'conv3', 'conv4'))
pylab.setp(ax[0], xticks=np.arange(len(cnn_layers)-1) + width / 2, xticklabels=('conv1', 'conv2', 'conv3', 'conv4'))

ax[1].bar(np.arange(1),[adacnn_mean_total],width,yerr=[adacnn_std_total],label='AdaCNN')
ax[1].bar(np.arange(1)+width,[fixed_total],width,label='Fixed-CNN')
ax[1].set_title('Total Parameters of CNN')
pylab.setp(ax[1], xticks=[], xticklabels=[])

ax[1].legend(bbox_to_anchor=(1.1,1))

pylab.subplots_adjust(wspace=0.3,right=0.8,top=0.85)
pylab.show()

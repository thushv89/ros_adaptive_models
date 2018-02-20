import matplotlib
import matplotlib.pyplot as pylab
import csv
import os
import numpy as np


log_dir = 'batch_norm_plots'
sub_log_dirs = ['no-batchnorm','batchnorm']

f, ax = pylab.subplots(1,3)

no_bn_x_axis = {0:None, 1:None, 2:None}
no_bn_soft_accuracy = {0:None, 1:None, 2:None}
no_bn_hard_accuracy = {0:None, 1:None, 2:None}

for env_idx in range(3):
    with open(log_dir + os.path.sep + sub_log_dirs[0] + os.path.sep + 'Error_'+str(env_idx)+'.log') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for r_i, row in enumerate(csvreader):
            if r_i != 0:
                main_epoch, env_epoch, hard_accuracy, soft_accuracy = int(row[0]),int(row[1]), float(row[3]),float(row[4])
                if no_bn_x_axis[env_idx] is None:
                    no_bn_x_axis[env_idx] = np.array(r_i*5)
                else:
                    no_bn_x_axis[env_idx] = np.append(no_bn_x_axis[env_idx],np.array(r_i*5))

                if no_bn_soft_accuracy[env_idx] is None:
                    no_bn_soft_accuracy[env_idx] = np.array(soft_accuracy)
                else:
                    no_bn_soft_accuracy[env_idx] = np.append(no_bn_soft_accuracy[env_idx],np.array(soft_accuracy))

                if no_bn_hard_accuracy[env_idx] is None:
                    no_bn_hard_accuracy[env_idx] = np.array(hard_accuracy)
                else:
                    no_bn_hard_accuracy[env_idx] = np.append(no_bn_hard_accuracy[env_idx],np.array(hard_accuracy))

bn_x_axis = {0: None, 1: None, 2: None}
bn_soft_accuracy = {0: None, 1: None, 2: None}
bn_hard_accuracy = {0: None, 1: None, 2: None}

for env_idx in range(3):
    with open(log_dir + os.path.sep + sub_log_dirs[1] + os.path.sep + 'Error_' + str(env_idx) + '.log') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for r_i, row in enumerate(csvreader):
            if r_i != 0:
                main_epoch, env_epoch, hard_accuracy, soft_accuracy = int(row[0]), int(row[1]), float(row[3]), float(
                    row[4])
                if bn_x_axis[env_idx] is None:
                    bn_x_axis[env_idx] = np.array(r_i*5)
                else:
                    bn_x_axis[env_idx] = np.append(bn_x_axis[env_idx], np.array(r_i*5))

                if bn_soft_accuracy[env_idx] is None:
                    bn_soft_accuracy[env_idx] = np.array(soft_accuracy)
                else:
                    bn_soft_accuracy[env_idx] = np.append(bn_soft_accuracy[env_idx], np.array(soft_accuracy))

                if bn_hard_accuracy[env_idx] is None:
                    bn_hard_accuracy[env_idx] = np.array(hard_accuracy)
                else:
                    bn_hard_accuracy[env_idx] = np.append(bn_hard_accuracy[env_idx], np.array(hard_accuracy))

for env_idx in range(3):

    ax[env_idx].plot(no_bn_x_axis[env_idx],no_bn_soft_accuracy[env_idx],color='r',label='without BN')
    #ax[env_idx].plot(no_bn_x_axis[env_idx], no_bn_hard_accuracy[env_idx], linestyle='--', color='r')
    ax[env_idx].set_xlabel('Iteration',fontsize=20)
    ax[env_idx].set_ylabel('Accuracy',fontsize=20)
    ax[env_idx].set_title('Environment %d'%env_idx,fontsize=24)
    ax[env_idx].plot(bn_x_axis[env_idx], bn_soft_accuracy[env_idx], color='b', label='with BN')
    #ax[env_idx].plot(bn_x_axis[env_idx], bn_hard_accuracy[env_idx], linestyle='--', color='b')

ax[2].legend(bbox_to_anchor=(1.35,1))
pylab.suptitle('Soft Accuracies in Three Environments (With BN vs W/O BN)', fontsize=24,y =0.98)
pylab.subplots_adjust(top=0.8,left=0.05,right=0.9,bottom=0.15)
pylab.show(block=True)
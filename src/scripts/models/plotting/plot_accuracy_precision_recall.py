import matplotlib
import matplotlib.pyplot as pylab
import csv
import os
import numpy as np


log_dir = 'accuracy_precision_recall'
sub_log_dirs = ['adacnn','rigid-pool','rigid-nonpool']

#f, ax = pylab.subplots(3,3)

algo_x_axes = [{0:None, 1:None, 2:None} for _ in range(3)]
algo_soft_accuracies = [{0:None, 1:None, 2:None} for _ in range(3)]
algo_precisions = [{0:None, 1:None, 2:None} for _ in range(3)]
algo_recalls = [{0:None, 1:None, 2:None} for _ in range(3)]

for alg_i in range(3):
    for env_idx in range(3):
        with open(log_dir + os.path.sep + sub_log_dirs[alg_i] + os.path.sep + 'Error_'+str(env_idx)+'.log') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for r_i, row in enumerate(csvreader):
                if r_i != 0:

                    main_epoch, env_epoch, soft_accuracy = int(row[0]),int(row[1]), float(row[4])
                    prec_l,prec_s,prec_r = float(row[6]), float(row[7]), float(row[8])
                    rec_l, rec_s, rec_r = float(row[10]), float(row[11]), float(row[12])

                    ## Building x axes for all algorithms all environments
                    if algo_x_axes[alg_i][env_idx] is None:
                        algo_x_axes[alg_i][env_idx] = np.array((r_i-1)*5)
                    else:
                        algo_x_axes[alg_i][env_idx] = np.append(algo_x_axes[alg_i][env_idx],np.array((r_i-1)*5))

                    if algo_soft_accuracies[alg_i][env_idx] is None:
                        algo_soft_accuracies[alg_i][env_idx] = np.array(soft_accuracy)
                    else:
                        algo_soft_accuracies[alg_i][env_idx] = np.append(algo_soft_accuracies[alg_i][env_idx],np.array(soft_accuracy))

                    if algo_precisions[alg_i][env_idx] is None:
                        algo_precisions[alg_i][env_idx] = np.array([prec_l,prec_s,prec_r]).reshape(1,-1)
                    else:
                        algo_precisions[alg_i][env_idx] = np.append(algo_precisions[alg_i][env_idx],
                                                                    np.array([prec_l,prec_s,prec_r]).reshape(1,-1),axis=0)

                    if algo_recalls[alg_i][env_idx] is None:
                        algo_recalls[alg_i][env_idx] = np.array([rec_l,rec_s,rec_r]).reshape(1,-1)
                    else:
                        algo_recalls[alg_i][env_idx] = np.append(algo_recalls[alg_i][env_idx],
                                                                    np.array([rec_l,rec_s,rec_r]).reshape(1,-1),axis=0)


linewidth=2
linewidth_prec = 1
linewidth_rec = 1
label_fontsize=14
alg_labels = ['AdaCNN','Fixed-CNN-B','Fixed-CNN']
line_styles = ['-','-','-']
colors = ['r','g','b']


# Accuracy plots
accuracy_axes = []
ax1 = pylab.subplot2grid((3, 9), (0, 0), colspan=3)
ax2 = pylab.subplot2grid((3, 9), (1, 0), colspan=3)
ax3 = pylab.subplot2grid((3, 9), (2, 0), colspan=3)
ax1.set_xlim(0,1500)
ax1.set_ylim(0,100)
ax1.set_ylabel('Environment 1',fontsize=label_fontsize)
ax2.set_xlim(0,1500)
ax2.set_ylim(0,100)
ax2.set_ylabel('Environment 2',fontsize=label_fontsize)
ax3.set_xlim(0,1500)
ax3.set_ylim(0,100)
ax3.set_ylabel('Environment 3',fontsize=label_fontsize)
ax3.set_xlabel('Iterations',fontsize=label_fontsize)
accuracy_axes.extend([ax1,ax2,ax3])

# Precision
precision_axes = []
for env_idx in range(3):
    precision_axes.append([None,None,None])
    precision_axes[env_idx][0] = pylab.subplot2grid((3, 9), (env_idx, 3))
    precision_axes[env_idx][1] = pylab.subplot2grid((3, 9), (env_idx, 4))
    precision_axes[env_idx][2] = pylab.subplot2grid((3, 9), (env_idx, 5))
    precision_axes[env_idx][0].set_yticks((0.0,1.0))
    precision_axes[env_idx][0].set_xticks((0,1500))
    precision_axes[env_idx][0].set_xlim(0,1500)
    precision_axes[env_idx][0].set_ylim(0, 1.0)
    precision_axes[env_idx][1].set_yticks((0.0,1.0))
    precision_axes[env_idx][1].set_xticks((0, 1500))
    precision_axes[env_idx][1].set_xlim(0, 1500)
    precision_axes[env_idx][1].set_ylim(0, 1.0)
    precision_axes[env_idx][2].set_yticks((0.0,1.0))
    precision_axes[env_idx][2].set_xticks((0, 1500))
    precision_axes[env_idx][2].set_xlim(0, 1500)
    precision_axes[env_idx][2].set_ylim(0, 1.0)

# Recall
recall_axes = []
for env_idx in range(3):
    recall_axes.append([None,None,None])
    recall_axes[env_idx][0] = pylab.subplot2grid((3, 9), (env_idx, 6))
    recall_axes[env_idx][1] = pylab.subplot2grid((3, 9), (env_idx, 7))
    recall_axes[env_idx][2] = pylab.subplot2grid((3, 9), (env_idx, 8))

    recall_axes[env_idx][0].set_yticks((0.0, 1.0))
    recall_axes[env_idx][0].set_xticks((0, 1500))
    recall_axes[env_idx][0].set_xlim(0,1500)
    recall_axes[env_idx][0].set_ylim(0, 1.0)
    recall_axes[env_idx][1].set_yticks((0.0, 1.0))
    recall_axes[env_idx][1].set_xticks((0, 1500))
    recall_axes[env_idx][1].set_xlim(0, 1500)
    recall_axes[env_idx][1].set_ylim(0, 1.0)
    recall_axes[env_idx][2].set_yticks((0.0, 1.0))
    recall_axes[env_idx][2].set_xticks((0, 1500))
    recall_axes[env_idx][2].set_xlim(0, 1500)
    recall_axes[env_idx][2].set_ylim(0, 1.0)


for env_idx in range(3):
    for alg_i in range(3):
        accuracy_axes[env_idx].plot(
            algo_x_axes[alg_i][env_idx],
            algo_soft_accuracies[alg_i][env_idx],
            color=colors[alg_i],label=alg_labels[alg_i],
            linestyle=line_styles[alg_i],
            linewidth=linewidth
        )



        # precision recall for each direction
        for di,direct in enumerate(['Left','Straight','Right']):
            precision_axes[env_idx][di].plot(
                algo_x_axes[alg_i][env_idx],
                algo_precisions[alg_i][env_idx][:,di],
                color=colors[alg_i], label=alg_labels[alg_i]+'-'+direct,
                linestyle=line_styles[alg_i],
                linewidth=linewidth_prec
            )

        for di,direct in enumerate(['Left','Straight','Right']):
            recall_axes[env_idx][di].plot(
                algo_x_axes[alg_i][env_idx],
                algo_recalls[alg_i][env_idx][:,di],
                color=colors[alg_i], label=alg_labels[alg_i]+'-'+direct,
                linestyle=line_styles[alg_i],
                linewidth=linewidth_rec
            )

    if env_idx==0:
        accuracy_axes[env_idx].set_title('Soft Accuracy')
        precision_axes[env_idx][0].set_title('Precision-Left')
        precision_axes[env_idx][1].set_title('Precision-Straight')
        precision_axes[env_idx][2].set_title('Precision-Right')
        recall_axes[env_idx][0].set_title('Recall-Left')
        recall_axes[env_idx][1].set_title('Recall-Straight')
        recall_axes[env_idx][2].set_title('Recall-Right')

    if env_idx == 2:
        precision_axes[env_idx][0].set_xlabel('Iterations',fontsize=label_fontsize)
        precision_axes[env_idx][1].set_xlabel('Iterations',fontsize=label_fontsize)
        precision_axes[env_idx][2].set_xlabel('Iterations',fontsize=label_fontsize)
        recall_axes[env_idx][0].set_xlabel('Iterations',fontsize=label_fontsize)
        recall_axes[env_idx][1].set_xlabel('Iterations',fontsize=label_fontsize)
        recall_axes[env_idx][2].set_xlabel('Iterations',fontsize=label_fontsize)
        #ax[env_idx].plot(no_bn_x_axis[env_idx], no_bn_hard_accuracy[env_idx], linestyle='--', color='r')
        #ax[env_idx][alg_i].set_xlabel('Iteration',fontsize=20)
        #ax[env_idx][alg_i].set_ylabel('Accuracy',fontsize=20)
        #ax[env_idx][alg_i].set_title('Environment %d'%env_idx,fontsize=24)

    #ax[env_idx].plot(bn_x_axis[env_idx], bn_hard_accuracy[env_idx], linestyle='--', color='b')

#pylab.suptitle('Soft Accuracies in Three Environments (With BN vs W/O BN)', fontsize=24,y =0.98)
pylab.subplots_adjust(top=0.95,left=0.04,right=0.97,bottom=0.15)
pylab.show(block=True)
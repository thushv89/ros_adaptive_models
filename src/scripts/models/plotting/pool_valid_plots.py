import csv
import os
import numpy as np
import matplotlib.pyplot as pylab
log_dir = 'pool_valid_plots'

pool_env_distribution = None
x_axis = []
with open(log_dir + os.path.sep + 'pool_valid_distribution.log') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for r_i, row in enumerate(csvreader):
        if r_i==0:
            continue

        main_epoch = int(row[0])
        train_env_idx = int(row[1])
        env_epoch = int(row[2])

        full_iteration = env_epoch + train_env_idx*100 + main_epoch*100*3
        x_axis.append(full_iteration)
        #print([row[-3], row[-2], row[-1]])
        curr_y = np.array([float(row[-4]), float(row[-3]), float(row[-2])]).reshape(1, -1)
        if pool_env_distribution is None:
            pool_env_distribution = curr_y
        else:
            pool_env_distribution = np.append(pool_env_distribution,curr_y,axis=0)

cum_pool_env_distribution = np.array(pool_env_distribution[:,0]).reshape(-1,1)
for env_idx in range(1,3):
    #print(cum_pool_env_distribution[:,env_idx-1])
    #print(pool_env_distribution[:,env_idx])
    cum_column = np.reshape(cum_pool_env_distribution[:,env_idx-1]+pool_env_distribution[:,env_idx],(-1,1))
    #print(cum_pool_env_distribution.shape)
    #print(cum_column.shape)
    cum_pool_env_distribution = np.append(cum_pool_env_distribution,cum_column,axis=1)

prev_line = np.zeros_like(cum_pool_env_distribution[:,0])
colors = ['r','g','b']
labels = ['Environment 1', 'Environment 2', 'Environment 3']
for env_idx in range(3):
    pylab.plot(x_axis, cum_pool_env_distribution[:,env_idx],color=colors[env_idx],label=labels[env_idx])

    pylab.fill_between(x_axis,cum_pool_env_distribution[:,env_idx],prev_line,facecolor=colors[env_idx])
    prev_line = cum_pool_env_distribution[:,env_idx]

for x in range(0,900,100):
    pylab.plot([x,x],[0,750.0],color='gray',linestyle='--')
pylab.ylim(0,750.0)
pylab.xlim(0,900.0)
pylab.xlabel('Iterations',fontsize=20)
pylab.ylabel('Amount of Data Present in the Pool,\n Belonging to a Single Environment',fontsize=20)
pylab.legend(fontsize=20,loc=2)
pylab.title('Data Distribution in $B_{pool}^{valid}$ Over Time',fontsize=22)
pylab.show(block=True)
import matplotlib
matplotlib.use('Qt5Agg')
print(matplotlib.get_backend())
import matplotlib.pyplot as pylab
pylab.switch_backend('Qt5Agg')
import csv
import os
import numpy as np


log_dir = 'utility_function'


#f, ax = pylab.subplots(3,3)

x_axis = []
q_values = None

actions = ['Add-1','Add-2','Add-3','Add-4','Rmv-1','Rmv-2','Rmv-3','Rmv-4','FT','NT','DN']

with open(log_dir + os.path.sep +  'predicted_q.log') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for r_i, row in enumerate(csvreader):
        if r_i != 0:
            x = int(row[0])
            x_axis.append(x)
            q = [float(q_comp) for q_comp in row[1:len(actions)+1]]
            if q_values is None:
                q_values = np.array(q).reshape(1,-1)
            else:
                q_values = np.append(q_values,np.array(q).reshape(1,-1),axis=0)

adapt_colors = ['r','g','b','y']
adapt_labels = [actions[ai] for ai in [0,3,4,7]]
global_colors = ['r','g','b']
global_labels = [actions[ai] for ai in [-3,-2,-1]]
adapt_q_values = q_values[:,[0,3,4,7]]
global_q_values = q_values[:,[-3,-2,-1]]

prev_x = x_axis[0]
prev_adapt_max = np.max(adapt_q_values[0,:])
prev_global_max = np.max(global_q_values[0,:])

f,ax = pylab.subplots(2,1)

seen_adapt_labels,seen_global_labels = [],[]

for x_i, x in enumerate(x_axis[1:]):

    ac_i = np.argmax(adapt_q_values[x_i,:])
    gc_i = np.argmax(global_q_values[x_i,:])

    if ac_i not in seen_adapt_labels:
        ax[0].plot([prev_x,x],[prev_adapt_max,np.max(adapt_q_values[x_i,:])],
                   color=adapt_colors[ac_i],label=adapt_labels[ac_i])
    else:
        ax[0].plot([prev_x, x], [prev_adapt_max, np.max(adapt_q_values[x_i, :])],
                   color=adapt_colors[ac_i])

    if gc_i not in seen_global_labels:
        ax[1].plot([prev_x,x],[prev_global_max,np.max(global_q_values[x_i,:])],
                   color=global_colors[gc_i],linewidth=2,label=global_labels[gc_i])
    else:
        ax[1].plot([prev_x, x], [prev_global_max, np.max(global_q_values[x_i, :])],
                   color=global_colors[gc_i], linewidth=2)

    seen_global_labels.append(gc_i)
    seen_adapt_labels.append(ac_i)

    prev_x = x
    prev_adapt_max = np.max(adapt_q_values[x_i,:])
    prev_global_max = np.max(global_q_values[x_i,:])
pylab.savefig('utility.png')
ax[0].legend()
ax[1].legend()
ax[1].set_ylabel('Maximum Q Value\nfor a given $s_t$')
ax[0].set_ylabel('Maximum Q Value\nfor a given $s_t$')
ax[1].set_xlabel('Iterations')
#pylab.interactive(False)
pylab.show(block=True)

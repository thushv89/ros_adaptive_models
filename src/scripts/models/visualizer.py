import matplotlib.pyplot as plt
import numpy as np

def save_fig_with_predictions_for_direction(predicted_img_ids, predicted_dict, filename):

    imgs_per_direction = 10
    fig, ax = plt.subplots(nrows=3,ncols=imgs_per_direction)

    for di,direct in enumerate(['left','straight','right']):
        if len(predicted_img_ids[direct]) == 0:
            continue
        perm_indices = np.random.permutation(predicted_img_ids[direct])
        for index in range(min(imgs_per_direction,len(predicted_img_ids[direct]))):

            norm_img = predicted_dict[perm_indices[index]][0] - np.min(predicted_dict[perm_indices[index]][0])
            norm_img = (norm_img/np.max(norm_img))
            ax[di,index].imshow(norm_img)
            ax[di,index].axis('off')
            ax[di,index].set_title(direct)

    fig.savefig(filename)
    plt.close(fig)
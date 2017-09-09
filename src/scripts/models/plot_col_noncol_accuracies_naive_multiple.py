#import matplotlib
#matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import numpy as np


vector_description = 'Epoch;Non-collision Accuracy;Non-collision Accuracy(Soft);Collision Accuracy;Collision Accuracy (Soft);Loss (Non-collision); Loss (collision); Precision-NC-L;Precision-NC-S;Precision-NC-R;;Recall-NC-L;Recall-NC-S;Recall-NC-R;;Precision-C-L;Precision-C-S;Precision-C-R;;Recall-C-L;Recall-C-S;Recall-C-R;'

multiple_summary_dictionary = {
    'apartment-my1-2000':'49;57.625;59.750;48.205;70.769;0.46878;-1.00000;0.615,0.591,0.526,;0.650,0.318,0.826,;0.491,0.298,0.570,;0.869,0.592,0.662,',
    'apartment-my2-2000':'49;61.448;64.570;62.143;75.000;0.50087;-1.00000;0.610,0.485,0.696,;0.788,0.514,0.635,;0.617,0.463,0.613,;0.630,0.792,0.826,',
    'apartment-my3-2000': '49;60.988;63.439;48.571;66.429;0.47701;-1.00000;0.629,0.527,0.608,;0.711,0.432,0.760,;0.456,0.339,0.510,;0.559,0.862,0.570,',
    'indoor-1-2000':'49;49.956;52.159;59.600;86.000;0.48044;-1.00000;0.501,0.356,0.599,;0.731,0.287,0.547,;0.689,0.351,0.584,;0.747,0.952,0.880,',
    'indoor-1-my1-2000':'49;51.553;57.816;49.722;72.222;0.47482;-1.00000;0.599,0.374,0.586,;0.606,0.526,0.603,;0.453,0.411,0.486,;0.717,0.600,0.850,',
    'grande_salle-my1-2000':'49;43.306;46.163;39.286;66.429;0.45908;-1.00000;0.453,0.346,0.470,;0.616,0.276,0.494,;0.398,0.337,0.432,;0.576,0.719,0.696,',
    'grande_salle-my2-2000':'49;52.328;56.638;41.786;57.143;0.48898;-1.00000;0.533,0.377,0.614,;0.752,0.387,0.559,;0.531,0.333,0.348,;0.652,0.469,0.598,',
}

naive_summary_dictionary = {
    'apartment-my1-2000':'49;32.333;99.958;33.143;33.143;0.42246;-1.00000;0.332;0.335;0.333;;0.999;1.000;1.000;;0.000;0.331;0.000;;0.000;1.000;0.000;',
    'apartment-my2-2000':'49;36.682;89.227;26.000;26.000;0.42121;-1.00000;0.333;0.336;0.333;;1.000;0.678;1.000;;0.000;0.260;0.000;;0.000;1.000;0.000;',
    'apartment-my3-2000':'49;32.960;96.320;33.600;33.600;0.42011;-1.00000;0.333;0.339;0.333;;1.000;0.890;1.000;;0.000;0.336;0.000;;0.000;1.000;0.000;',
    'indoor-1-2000':'49;35.556;100.000;33.600;33.600;0.42188;-1.00000;0.334;0.332;0.334;;1.000;1.000;1.000;;0.000;0.336;0.000;;0.000;1.000;0.000;',
    'indoor-1-my1-2000':'49;39.366;97.122;32.000;32.000;0.42159;-1.00000;0.333;0.338;0.333;;1.000;0.914;1.000;;0.000;0.320;0.000;;0.000;1.000;0.000;',
    'grande_salle-my1-2000':'49;33.469;100.000;36.000;36.000;0.41778;-1.00000;0.333;0.335;0.333;;1.000;1.000;1.000;;0.000;0.360;0.000;;0.000;1.000;0.000;',
    'grande_salle-my2-2000':'49;31.696;99.783;33.600;33.600;0.41898;-1.00000;0.333;0.335;0.333;;1.000;0.993;1.000;;0.000;0.336;0.000;;0.000;1.000;0.000',
    'sandbox-2000':'49;33.879;100.000;36.000;36.000;0.41990;-1.00000;0.333;0.333;0.333;;1.000;1.000;1.000;;0.000;0.360;0.000;;0.000;1.000;0.000'
}


def clean_string_and_return_elements(desc_string,dtype):
    descriptions = []

    # make all the delimiters semicolons. Then easier to tokenize
    desc_string = desc_string.replace(',',';')
    str_tokens = desc_string.split(';')

    for tok in str_tokens:
        if len(tok)==0:
            continue

        tok = tok.strip()
        for char in tok:
            if char==',' or char==';':
                del char
        if dtype == 'int':
            descriptions.append(int(tok))
        elif dtype == 'float':
            descriptions.append(float(tok))
        elif dtype=='string':
            descriptions.append(str(tok))
        else:
            raise NotImplementedError

    return descriptions

def clean_string_vectors(multi_summary,naive_summary,desc_elements):
    multi_arranged_dict,naive_arranged_dict = {},{}
    for multi_k,naive_k in zip(multi_summary.keys(),naive_summary.keys()):
        multi_str_elements = clean_string_and_return_elements(multi_summary[multi_k],'float')
        naive_str_elements = clean_string_and_return_elements(naive_summary[naive_k],'float')

        multi_arranged_dict[multi_k] = {}
        naive_arranged_dict[naive_k] = {}
        for mul,nai,desc in zip(multi_str_elements,naive_str_elements,desc_elements):
            multi_arranged_dict[multi_k][desc] = mul
            naive_arranged_dict[naive_k][desc] = nai

    return multi_arranged_dict,naive_arranged_dict

def get_column_plot_with_subset(ax, environment, multi_summary,naive_summary,sub_titles,sub_y_labels,sub_x_labels, title,fontsize,title_fontsize):
    num_sec = len(sub_titles)
    col_width = 0.2

    ax.set_ylabel(sub_y_labels, fontsize=fontsize)
    ax.set_title(title,fontsize=title_fontsize)


    multi_data, naive_data = [], []
    for idx, _ in enumerate(sub_titles):
        mul_sample = multi_summary[environment][sub_titles[idx]]
        nai_sample = naive_summary[environment][sub_titles[idx]]
        mul_sample = mul_sample if mul_sample>0 else 0.0001
        nai_sample = nai_sample if nai_sample>0 else 0.0001
        multi_data.append(mul_sample)
        naive_data.append(nai_sample)

    bar_indices = np.arange(0, len(sub_titles))

    ax.set_xticks(bar_indices+ col_width)
    ax.set_xticklabels(sub_x_labels)

    mult_rects = ax.bar(bar_indices, multi_data, col_width, color='r')
    naiv_rects = ax.bar(bar_indices + col_width, naive_data, col_width, color='b')

    return mult_rects,naiv_rects


if __name__ == '__main__':

    desc_elements = clean_string_and_return_elements(vector_description,'string')
    print(desc_elements)
    multi_dict, naive_dict = clean_string_vectors(multiple_summary_dictionary,naive_summary_dictionary,desc_elements)

    print(multi_dict.keys())
    print(naive_dict.keys())
    print(multi_dict[multi_dict.keys()[0]])
    fig, axarr = plt.subplots(nrows=1, ncols=3)

    environment = 'apartment-my2-2000'
    ax_1sub_titles = ['Non-collision Accuracy', 'Non-collision Accuracy(Soft)',
                                 'Collision Accuracy','Collision Accuracy (Soft)']
    ax1_sub_x_labels = ['NC (Hard)', 'NC (Soft)',
                                 'C (Hard)','C (Soft)']
    ax1_sub_y_labels = 'Test Accuracy (%)'
    fontsize = 18

    main_title = "CerberusCNN vs Vanilla CNN Performance\n(Apartment-2 Test Dataset)"
    mul_rects, nai_rects = get_column_plot_with_subset(axarr[0], environment, multi_dict, naive_dict, ax_1sub_titles, ax1_sub_y_labels, ax1_sub_x_labels, 'Collision and Non-Collision Accuracies', fontsize, fontsize)
    axarr[0].plot(np.arange(5),[33.3 for _ in range(5)],c='gray')

    '''Precision - NC - L;
    Precision - NC - S;
    Precision - NC - R;;Recall - NC - L;
    Recall - NC - S;
    Recall - NC - R;;Precision - C - L;
    Precision - C - S;
    Precision - C - R;;Recall - C - L;
    Recall - C - S;
    Recall - C - R'''

    ax2_sub_titles = ['Precision-NC-L','Precision-NC-S','Precision-NC-R']
    ax2_x_labels = ['NC-Left', 'NC-Straight', 'NC-Right']
    ax2_y_labels = ''
    get_column_plot_with_subset(axarr[1],environment,multi_dict,naive_dict,ax2_sub_titles,ax2_y_labels,ax2_x_labels,'Non-Collision Precision',fontsize,fontsize)

    ax3_sub_titles = ['Precision-C-L', 'Precision-C-S', 'Precision-C-R']
    ax3_x_labels = ['C-Left', 'C-Straight', 'C-Right']
    ax3_y_labels = ''
    get_column_plot_with_subset(axarr[2], environment, multi_dict, naive_dict, ax3_sub_titles, ax3_y_labels,
                                ax3_x_labels, 'Collision Precision', fontsize, fontsize)

    fig.subplots_adjust(bottom=0.05, top=0.92, left=0.07, right = 0.97)
    plt.show()

# import tensorflow as tf
# from keras.models import load_model
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib_venn import venn3_unweighted, venn3_circles


if __name__ == '__main__':

    # tf.keras.backend.clear_session()

    with open('/Users/dmitryduev/_caltech/python/deep-asteroids/service/code/config.json') as f:
        config = json.load(f)

    models = config['models']
    model_names = list(models.keys())

    path_models = '/Users/dmitryduev/_caltech/python/deep-asteroids/service/models'

    path_logs = '/Users/dmitryduev/_caltech/python/deep-asteroids/paper'

    ''' training/validation accuracies '''

    c_families = ('rb', 'sl', 'kd', 'os')

    for c_family in c_families:

        print(c_family)
        mn = [m_ for m_ in model_names if c_family in m_]

        acc = dict()
        val_acc = dict()

        fig = plt.figure(figsize=(9, 3))
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15, left=0.080, right=0.960)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        for model_name in mn:

            # model_base_name = os.path.join(path_models, models[model_name])
            #
            # print(f'loading {model_name}')
            # m = load_model(f'{model_base_name}.h5')

            data_set = models[model_name].split('_')[0]
            # print(data_set)

            path_data_set = os.path.join(path_logs, data_set)

            path_acc = os.path.join(path_data_set, f'run_{models[model_name]}-tag-acc.json')
            path_val_acc = os.path.join(path_data_set, f'run_{models[model_name]}-tag-val_acc.json')

            acc[model_name] = pd.read_json(path_acc)
            val_acc[model_name] = pd.read_json(path_val_acc)

            # print(acc[model_name])
            ax1.plot(acc[model_name][2], '-', linewidth=1.8, markersize=3.0, label=model_name)
            ax2.plot(val_acc[model_name][2], '-', linewidth=1.8, markersize=3.0, label=model_name)

        # ax1.set_ylim([0.63, 1.02])
        ax1.set_ylim([0.89, 1.02])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Accuracy')
        ax1.legend(loc='best')
        ax1.grid(True)

        # ax2.set_ylim([0.63, 1.02])
        ax2.set_ylim([0.72, 1.02])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy')
        ax2.legend(loc='best')
        ax2.grid(True)

        # plt.show()
        fig.savefig(f'/Users/dmitryduev/_caltech/python/deep-asteroids/paper/{c_family}_acc.png', dpi=300)

    ''' Real zoo '''
    path_reals = '/Users/dmitryduev/_caltech/python/deep-asteroids/paper/reals_201812_201901'

    # fig2 = plt.figure(figsize=(10, 4.5))
    fig2 = plt.figure()
    fig2.subplots_adjust(top=1.0, bottom=0.0, left=0.015, right=0.99, hspace=0.0, wspace=0.03)
    # fig2 = plt.figure()

    for ii, fr in enumerate(glob.glob(os.path.join(path_reals, '*.jpg'))):
        ax_ = fig2.add_subplot(3, 8, ii+1)
        ax_.imshow(mpimg.imread(fr), interpolation='nearest')
        ax_.axis('off')

    ''' Venn diagrams '''
    fig3 = plt.figure()
    fig3.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, hspace=0.2, wspace=0.2)
    # 2019/01/15: total 6312572;
    # rb 356304; sl 5819505; kd 149863; rb+sl 219660; rb+kd 39010; sl+kd 126308; rb+sl+kd 19210
    # v = venn3(subsets=(356304, 5819505, 219660, 149863, 39010, 126308, 19210),
    #           set_labels=('rb > 0.9', 'sl > 0.5', 'kd > 0.5'))
    # c = venn3_circles(subsets=(356304, 5819505, 219660, 149863, 39010, 126308, 19210),
    #                   linestyle='dashed', linewidth=1, color="grey")
    v = venn3_unweighted(subsets=(356304, 5819505, 219660, 149863, 39010, 126308, 19210),
                         set_labels=('rb > 0.9', 'sl > 0.5', 'kd > 0.5'))
    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed', linewidth=1, color="grey")

    fig3.savefig('/Users/dmitryduev/_caltech/python/deep-asteroids/paper/venn3_rb_sl_kd.png', dpi=300)

    # plt.show()

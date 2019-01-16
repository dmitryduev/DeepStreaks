import os
import json
import glob
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def load_data(path: str = './data', project_id: str = None, binary: bool = True, grayscale: bool = True,
              resize: tuple = (144, 144), test_size=0.1, verbose: bool = True, random_state=None):

    # data:
    x = []
    # classifications:
    y = []

    # get json file with project metadata
    project_meta_json = os.path.join(path, f'{project_id}.json')
    with open(project_meta_json) as f:
        project_meta = json.load(f)

    # print(project_meta)

    classes_list = project_meta['classes']

    # if it's a binary problem, do {'class1': 0, 'class2': 1}
    # if multi-class (N), do {'class1': np.array([1, 0, ..., 0]), 'class2': np.array([0, 1, ..., 0]),
    #                         'classN': np.array([0, 0, ..., 1])}
    if binary:
        classes = {classes_list[0]: 0, classes_list[1]: 1}
    else:
        classes = {}
        n_c = len(classes_list)

        for ci, cls in enumerate(classes_list):
            classes[cls] = np.zeros(n_c)
            classes[cls][ci] = 1

    # print(classes)

    path_project = os.path.join(path, project_id)

    for dataset_id in project_meta['datasets']:
        if verbose:
            print(f'Loading dataset {dataset_id}')

        dataset_json = sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*.json')))
        if len(dataset_json) > 0:
            dataset_json = dataset_json[-1]

            with open(dataset_json) as f:
                classifications = json.load(f)
            # print(classifications)

            path_dataset = sorted(glob.glob(os.path.join(path_project, f'{dataset_id}.*')))[0]
            # print(path_dataset)

            for k, v in classifications.items():
                # resize and normalize:
                image_path = os.path.join(path_dataset, k)

                if os.path.exists(image_path):

                    if grayscale:
                        # the assumption is that images are grayscale
                        img = np.array(ImageOps.grayscale(Image.open(image_path)).resize(resize, Image.BILINEAR)) / 255.
                        img = np.expand_dims(img, 2)

                    else:
                        # make it rgb:
                        img = ImageOps.grayscale(Image.open(image_path)).resize(resize, Image.BILINEAR)
                        rgbimg = Image.new("RGB", img.size)
                        rgbimg.paste(img)
                        img = np.array(rgbimg) / 255.

                    x.append(img)

                    image_class = classes[v[0]]
                    y.append(image_class)

    # numpy-fy and split to test/train

    x = np.array(x)
    y = np.array(y)

    if verbose:
        print(x.shape)
        print(y.shape)

        # check statistics on different classes
        if not binary:
            print('\n')
            for i, ii in enumerate(sorted(classes.keys())):
                print(f'{ii}:', np.sum(y[:, i]))
            print('\n')
        else:
            print('\n')
            cs = list(classes.keys())
            print(f'{cs[0]}:', len(y) - np.sum(y))
            print(f'{cs[1]}:', np.sum(y))
            print('\n')

    if random_state is None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    if verbose:
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    return x_train, y_train, x_test, y_test, classes
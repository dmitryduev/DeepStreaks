import json
import os
import glob
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def load_data(path: str = './data', project_ids: dict = {'os': '5c05bbdc826480000a95c0bf'},
              binary: bool = True, grayscale: bool = True,
              resize: tuple = (144, 144), test_size=0.1, verbose: bool = True, random_state: int = 42):

    files = dict()

    for classifier in project_ids.keys():
        if verbose:
            print(f'Loading project {classifier}: {project_ids[classifier]}')

        # file names:
        x = []
        # classifications:
        y = []

        # get json file with os project metadata
        project_meta_json = os.path.join(path, f'{project_ids[classifier]}.json')
        with open(project_meta_json) as f:
            project_meta = json.load(f)

        path_project = os.path.join(path, project_ids[classifier])

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
                    image_path = os.path.join(path_dataset, k)

                    if os.path.exists(image_path):

                        x.append(k)
                        y.append(1)

        # numpy-fy and split to test/train
        # os 2648
        # rb 2531
        # sl 1642
        # kd 2206

        x = np.array(x)
        y = np.array(y)

        x_train, x_test, _, _ = train_test_split(x, y, test_size=test_size, random_state=random_state)

        if classifier == 'os':
            files[classifier] = set(x_test)
        else:
            files[classifier] = set(x_train)

    for classifier in files:
        print(classifier, len(files[classifier]))

    ''' Load test data for os classifiers that has not been used in training for other classifiers '''

    # data:
    x = []
    # classifications:
    y = []

    # get json file with project metadata
    project_meta_json = os.path.join(path, f'{project_ids["os"]}.json')
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

    path_project = os.path.join(path, project_ids['os'])

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

                if os.path.exists(image_path) and (k in files['os']) and \
                        (k not in files['rb']) and (k not in files['sl']) and (k not in files['kd']):

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

    # load reals not used in training:
    path_reals = './data-raw/reals_20181201_20190124'
    reals = glob.glob(os.path.join(path_reals, '*.jpg'))

    for image_path in reals:

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

            image_class = 1
            y.append(image_class)

    # numpify

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

    return x, y


if __name__ == '__main__':

    x_, y_ = load_data(project_ids={'os': '5c05bbdc826480000a95c0bf',
                                    'rb': '5b96af9c0354c9000b0aea36',
                                    'sl': '5b99b2c6aec3c500103a14de',
                                    'kd': '5be0ae7958830a0018821794'})

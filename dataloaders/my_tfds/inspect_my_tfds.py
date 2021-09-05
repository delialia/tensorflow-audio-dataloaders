""" Some prints to check the tfds dataset
    For the official documentation on how to test it refer to :
    https://www.tensorflow.org/datasets/add_dataset
"""
# external
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def get_count(ds):
    labels, counts = np.unique(np.fromiter(ds.map(lambda x, y: y), float), return_counts=True)
    ind_sort = np.argsort(-counts)
    return labels[ind_sort],counts[ind_sort]

def run():

    ds, info = tfds.load('test_dataset', with_info=True, as_supervised = True)
    print('-----------------------------------------------------')
    print(' INFO ')
    print(info)
    print('-----------------------------------------------------')
    print(' DATASET ')
    print(ds)
    print('-----------------------------------------------------')
    print(' LABEL INFO ')
    print(info.features["label"].num_cgitgit lasses)
    print(info.features["label"].names)
    #label distribution
    print('--> Label distribution in TRAIN SPLIT')
    for val, count in zip(*get_count(ds['train'])):
        print(info.features["label"].int2str(int(val)), count)
    print('--> Label distribution in TEST SPLIT')
    for val, count in zip(*get_count(ds['test'])):
        print(info.features["label"].int2str(int(val)), count)
    print('--> Label distribution in VAL SPLIT')
    for val, count in zip(*get_count(ds['val'])):
        print(info.features["label"].int2str(int(val)), count)

    # if dataset small can also do
    # print('-----------------------------------------------------')
    # print(' DATAFRAME ')
    # print(tfds.as_dataframe(ds['train'], info))
    # print('-----------------------------------------------------')
    # print(' LABEL COUNTS ')
    # print(tfds.as_dataframe(ds['train'], info).label.value_counts())




if __name__ == '__main__':
    run()

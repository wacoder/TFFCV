#!/usr/bin/ python3.5
#title           :convert_to_tf.py
#description     :This will convert the image dataset to tfrecords.
#author          :Jing Wang, Compass Digital Labs
#date            :20180319
#version         :1.1
#usage           :python pyscript.py
#notes           :
#python_version  :3.5.2 
#==============================================================================

# load library 
import argparse
import math 

import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import cv2

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, base_path, image_path, i):
    file_path = dataset.full_path
    label = dataset.label
    data_size = dataset.shape[0]
    tfrecords_path = os.path.join(base_path, "tfrecords")
    if not os.path.exists(tfrecords_path):
        os.mkdir(tfrecords_path)
    
    filename = os.path.join(tfrecords_path, "tfrecord"+'-%.3d'%i+'.tfrecords')

    print("Writing", filename)
    error = 0
    total = 0
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(data_size):
            try: 
                image = cv2.imread(os.path.join(image_path, str(file_path[index])), cv2.IMREAD_COLOR)
                image_raw = image.tostring()

            except IOError:
                error += 1
                pass
            if index % 1000 == 0:
                print("stps",index) 
            example = tf.train.Example(features= tf.train.Features(feature={
                'label' : _int64_feature(label[index]),
                'image' : _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())

            total += 1
    print("There are ", error, "missing pictures")
    print("Total ", total, 'valid pictures')



    


def get_meta(label_path):
    lb_encoder = LabelEncoder()
    meta = pd.read_csv("../data/trainLabels.csv")
    meta['full_path'] = meta.apply(lambda row: str(row['id'])+'.png', axis=1)
    meta['en_label'] = lb_encoder.fit_transform(meta['label'])

    data = {'full_path':meta['full_path'], 'label': meta['en_label']}
    dataset = pd.DataFrame(data)
    return dataset


def main(base_path, image_path, label_path, nworks):
    start_time = time.time()
    file_names = os.listdir(image_path)

    dataset_size = file_names.shape[0]
    index = np.linspace(0, dataset_size, nworks+1, dtype=np.int32)
    print(dataset_size)
    # pool = multiprocessing.Pool(processes=nworks)
    # for p in range(nworks):
    #     pool.apply_async(convert_to, (dataset[index[p]:index[p+1]].copy().reset_index(drop=True), base_path, image_path, p))
    # pool.close()
    # pool.join()

    # convert_to(dataset, base_path, image_path, 1)

    duration = time.time()-start_time

    print("Running {:.3f} sec".format(duration))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default='/workspace/docker/cifar/data')
    parser.add_argument("--label_path", type=str, default='/workspace/docker/cifar/data/trainLabels.csv')
    parser.add_argument("--image_path", type=str, default='/workspace/docker/cifar/data/train/train')
    parser.add_argument("--nworks", type=int, default=5)

    args = parser.parse_args()

    main(args.base_path, args.image_path, args.label_path, args.nworks)


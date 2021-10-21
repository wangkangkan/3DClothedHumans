#!/usr/bin/python
'''
    FlyingThings3D data preprocessing.
'''

import numpy as np
import os
import re
import sys
from scipy.io import loadmat
from read_depth import ImageCoder, sample_from_depth
from write2obj import read_obj

import argparse

import natsort
import glob

import tensorflow as tf

import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../datatran_10.0/alldepthfiles_100frm_4w8.txt', type=str, help='input root dir')
parser.add_argument('--output_dir', default='../hks_part327_full/data_preprocessing/faust', type=str, help='output dir')
# parser.add_argument('--output_dir', default='../flownet3d/doc/male_new', type=str, help='output dir')
FLAGS = parser.parse_args()


##############################################################################################################################################################

def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to_example(frame1, frame2, flow, frm1_point=6890, frm2_point=20000):
    """Build an Example proto for an image example.
    Args:
      frame1: 
      frame2:
      flow: label

    Returns:
      Example proto
    """

    frame1 = np.reshape(frame1, [frm1_point, 3])

    frame2 = np.reshape(frame2, [frm2_point, 3])
    flow = np.reshape(flow, [160])


    feat_dict = {
        'frame1/x': float_feature(frame1[:, 0].astype(np.float)),
        'frame1/y': float_feature(frame1[:, 1].astype(np.float)),
        'frame1/z': float_feature(frame1[:, 2].astype(np.float)),
        'frame2/x': float_feature(frame2[:, 0].astype(np.float)),
        'frame2/y': float_feature(frame2[:, 1].astype(np.float)),
        'frame2/z': float_feature(frame2[:, 2].astype(np.float)),
        'flow/x': float_feature(flow[:].astype(np.float)),
    }
    

    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))

    return example




def proc_one_scene_human():
    
    data_cate = "data5"
    train_or_test = 'train'
    output_dir = "../displacement/" + data_cate + "/tfrecord_stage1_template"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(output_dir, train_or_test + "_00032_" + '%03d.tfrecord')

    # coder = ImageCoder()
    frm2_num_point = 20000

    all_theta = np.loadtxt("../displacement/{}/thetafile_all.txt".format(data_cate))
    
    frm1_gt_list = natsort.natsorted(glob.glob('../displacement/{}/6890/*.txt'.format(data_cate)))
    frm2_pointcloud_list = natsort.natsorted(glob.glob('../displacement/{}/{:d}/*.txt'.format(data_cate, frm2_num_point)))
    fidx = 0
    seqnum = len(frm1_gt_list)
    print(seqnum)
    i = 0
    tnum_shards = 5000
    # sess = tf.Session()
    while i < seqnum:
        # Open new TFRecord file.
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < seqnum and j < tnum_shards:

                xyz2dir = frm1_gt_list[i]
                print(xyz2dir)


                frame1 = np.loadtxt(xyz2dir)
                frame2 = np.loadtxt(frm2_pointcloud_list[i])
                # stage1, _ = read_obj(stage1_list[i])
                print('Converting image %d/%d' % (i, seqnum))


                example = convert_to_example(frame1, frame2, all_theta[i], frm2_point=frm2_num_point) 
                                                # vector1, vector2, value1, value2)
                writer.write(example.SerializeToString())

                j += 1
                i += 1


        fidx += 1



def read_file_list(filelist):
    """
    Scan the image file and get the image paths and labels
    """
    with open(filelist) as f:
        lines = f.readlines()
        files = []  
        for l in lines:
            items = l.split()
            # print(items)
            # exit()
            files.append(items[0])
            #self.imagefiles.append(l)

        # store total number of data
    filenum = len(files)
    print("Training sample number: %d" % (filenum))
    return files


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


    
    proc_one_scene_human()

from os import makedirs
from os.path import join, exists
import numpy as np
from glob import glob
import pickle as pickle

import tensorflow as tf

from common import convert_to_example_our_3dpoint, ImageCoder,convert_to_example_our_correspondence
tf.app.flags.DEFINE_string(
    'dataset_name', 'neutrSMPL_jointLim',
    'neutrSMPL_CMU, neutrSMPL_H3.6, or neutrSMPL_jointLim')
tf.app.flags.DEFINE_string('data_directory',
                           'G:/endtoendtrainingdata/neutrMosh/',
                           'data directory where SMPL npz/pkl lies')
tf.app.flags.DEFINE_string('output_directory',
                           './tran_tfrecords_2_ft/',
                           'Output data directory')
#/scratch1/projects/tf_datasets/mocap_neutrMosh/
tf.app.flags.DEFINE_integer('num_shards', 10000,
                            'Number of shards in TFRecord files.')

FLAGS = tf.app.flags.FLAGS
fx_d = 3.6667199999999998e+002
cx_d = 2.5827199999999998e+002
fy_d = 3.6667199999999998e+002
cy_d = 2.0560100000000000e+002

def _add_to_tfrecord_3dpoints(image_path, label_path, initpara, coder, writer):
    with tf.gfile.FastGFile(image_path, 'rb') as f:
        depth_imagedata = f.read()

    depthimage = coder.decode_png16(depth_imagedata)
    height, width = depthimage.shape[:2]

    imx = np.tile(np.arange(width), (height, 1))
    imy = np.tile(np.reshape(np.arange(height), [-1, 1]), (1, width))
    imxlist = np.reshape(imx, [-1])
    imylist = np.reshape(imy, [-1])
    depthlist = np.reshape(np.array(depthimage, dtype=np.float32), [-1]) / 1000.0
    imxlist = (imxlist - cx_d) / fx_d * depthlist
    imylist = (imylist - cy_d) / fy_d * depthlist

    pidx = np.reshape(np.where(depthlist > 0), [-1])
    px = np.reshape(imxlist[pidx], [-1, 1])
    py = np.reshape(imylist[pidx], [-1, 1])
    pz = np.reshape(depthlist[pidx], [-1, 1])
    pointset = np.concatenate([px, py, pz], 1)

    pointcenter = np.mean(pointset,axis=0)
    pointset = pointset - np.tile(pointcenter, (pointset.shape[0], 1))

    label = np.loadtxt(label_path)
    label -= pointcenter
    modelvis = np.ones(6890)
    label = np.concatenate([label, modelvis], axis=1)

    num_sample = 2500#6890
    pnum = pointset.shape[0]
    if pnum>=1:
        if (pnum == num_sample):
            sidx = range(num_sample)
        elif (pnum > num_sample):
            sidx = np.random.choice(pnum, num_sample)
        else:
            sample = np.random.choice(pnum, num_sample - pnum)
            sidx = np.concatenate([range(pnum),sample],0)

        samplepointset = pointset[sidx,:]
    else:
        samplepointset = np.zeros((num_sample,3))

    example = convert_to_example_our_3dpoint(samplepointset,  label, initpara)
    writer.write(example.SerializeToString())

def process_ourdata(allimagefiles, alllabelfiles, initparas, num_shards, out_dir):

    coder = ImageCoder()
    out_path = join(out_dir, 'train_%03d.tfrecord')
    i = 0
    fidx = 0
    while i < len(allimagefiles):
        # Open new TFRecord file.
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < len(allimagefiles) and j < num_shards:
                # if i % 5 == 0:
                print('Converting image %d/%d' % (i, len(allimagefiles)))
                _add_to_tfrecord_3dpoints(
                allimagefiles[i],
                alllabelfiles[i],#labels[i, :, :],
                initparas[i,:],
                coder,
                writer)
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
            files.append(items[0])
            #self.imagefiles.append(l)

        # store total number of data
    filenum = len(files)
    print("Training sample number: %d" % (filenum))
    return files

def main(unused_argv):

    #depth images
    allimagefiles = read_file_list('/test/DFAUS/CAPE/cape_release/male/depthfilenamelist1.txt')
    alllabelfiles = read_file_list('/test/DFAUS/CAPE/cape_release/male/corfilenamelist1.txt')
    initparas = np.loadtxt('./allparameters_init_ft2500_p2.txt', dtype=np.float32)

    print('Saving results to %s' % FLAGS.output_directory)

    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)

    process_ourdata(allimagefiles, alllabelfiles, initparas, FLAGS.num_shards, FLAGS.output_directory)

if __name__ == '__main__':
    tf.app.run()

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

import config_our_3dpoint
from RunModel_our_3dpoint_ournetwork import RunModel

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

def main():
    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    sess = tf.Session(config=configs)
    model = RunModel(config, sess=sess)
    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002
    allimagefiles = read_file_list('/test/multistage/realdata/realdatatarget2.txt')
    pname = './realdata2_p2.txt'
    initparas = np.loadtxt('../stage1_new/realdata2_p2.txt', dtype=np.float32)
    f = open(pname, "a+")
    idx = 0
    sampleidx = 0
    while sampleidx < len(allimagefiles):
        # if sampleidx % 10 == 0:#5 20w  10 10w
        print('sample %d' % sampleidx)
       
        pointset = np.loadtxt(allimagefiles[sampleidx])
        pointcenter = np.mean(pointset, axis=0)
        pointset = pointset - np.tile(pointcenter, (pointset.shape[0], 1))

        num_sample = 2500
        pnum = pointset.shape[0]
        if pnum >= 1:
            if (pnum == num_sample):
                sidx = range(num_sample)
            elif (pnum > num_sample):
                sidx = np.random.choice(pnum, num_sample)
            else:
                sample = np.random.choice(pnum, num_sample - pnum)
                sidx = np.concatenate([range(pnum), sample], 0)

            samplepointset = pointset[sidx, :]
        else:
            samplepointset = np.zeros((num_sample, 3))

        samplepointset = samplepointset.reshape([1, num_sample, 3])

        verts, theta, cams_trl = model.predict_T(samplepointset, initparas[sampleidx].reshape([1,-1]))
        verts = np.squeeze(verts, 0)
        verts = verts + np.tile(pointcenter, (verts.shape[0], 1))
        model.savemodel_T(verts, pointcenter, sampleidx)

        for i in range(theta.shape[1]):
            f.write(str(theta[0,i]) + " ")
        f.write("\n")

        sampleidx += 1
    f.close()
    #model.savemodel(verts)
if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = config_our_3dpoint.PRETRAINED_MODEL
    config.batch_size = 1
    main()


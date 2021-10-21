import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

import config_our_3dpoint
from RunModel_our_3dpoint_ournetwork import RunModel

#     return crop, proc_param, img
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

def main(img_path, labelimg_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002
    allimagefiles = ["/test/DFAUS/realdata/kongfu/im156.png", ]

    sampleidx = 0
    while sampleidx < len(allimagefiles):
        # if sampleidx % 5 == 0:
        print('sample %d' % sampleidx)
        # path = '/test'+allimagefiles[sampleidx].split(':')[1]
        depthimage = io.imread(allimagefiles[sampleidx])
        # input_img = img.reshape([1, img.shape[0],img.shape[1], 1])
        # input_img = input_img.astype(np.float32)

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
        rawpointset = pointset
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

        theta, verts = model.predict_dict(samplepointset)

        model.savemodel(verts)

        sampleidx += 1
    # f.close()

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = config_our_3dpoint.PRETRAINED_MODEL

    config.batch_size = 1
    main(config.img_path, config.labelimg_path, config.json_path)

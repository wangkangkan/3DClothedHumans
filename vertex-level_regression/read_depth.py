import tensorflow as tf 
import numpy as np
import os
import argparse
import cv2
# from evaluatetest import read_file_list
import glob
from write2obj import read_obj
parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', default='/rendermale/depth/smpltrain0male_01_01_c0001_0_model_-1.png')
flags = parser.parse_args()

def gene_label(alllabelfiles, num_sample=3000):

    
    allimagefiles = alllabelfiles.replace('testmodelcor', 'testdepth')
    allimagefiles = allimagefiles.replace('_cor', '')

    depthimage = cv2.imread(allimagefiles,-1)
    # print(type(depthimage))
    # exit()

    height, width = depthimage.shape[:2]
    
    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002

    imx = np.tile(np.arange(width), (height, 1))
    imy = np.tile(np.reshape(np.arange(height), [-1, 1]), (1, width))
    imxlist = np.reshape(imx, [-1])
    imylist = np.reshape(imy, [-1])
    depthlist = np.reshape(np.array(depthimage, dtype=np.float32), [-1]) / 1000.0
    imxlist = (imxlist - cx_d) / fx_d * depthlist
    imylist = (imylist - cy_d) / fy_d * depthlist


    ##   depth ##################################
    pidx = np.reshape(np.where(depthlist > 0), [-1])
    px = np.reshape(imxlist[pidx], [-1, 1])
    py = np.reshape(imylist[pidx], [-1, 1])
    pz = np.reshape(depthlist[pidx], [-1, 1])
    pointset = np.concatenate([px, py, pz], 1)

    # num_sample = sample_num

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



    # cor = join('../hm36cor/', alllabelfiles[i])

    ##   label ###################################
    labelimage = cv2.imread(alllabelfiles,-1)
    labellist = np.reshape(np.array(labelimage, dtype=np.int32), [-1]) - 1
    select = np.reshape(np.where(labellist > -1), [-1])
    x = np.reshape(imxlist[select], [-1])
    y = np.reshape(imylist[select], [-1])
    z = np.reshape(depthlist[select], [-1])
    vidx = np.reshape(labellist[select], [-1])
    
    modelx = np.zeros(6890)
    modely = np.zeros(6890)
    modelz = np.zeros(6890)
    modelvis = np.zeros(6890)
    modelx[vidx] = x
    modely[vidx] = y
    modelz[vidx] = z
    nvis = np.array(np.greater(vidx, -1), np.float32)
    modelvis[vidx] = nvis
    
    modelx = np.reshape(modelx, [-1, 1])
    modely = np.reshape(modely, [-1, 1])
    modelz = np.reshape(modelz, [-1, 1])
    modelvis = np.reshape(modelvis, [-1, 1])
    modelxyz = np.concatenate([modelx, modely, modelz], 1)
    # modelxyz = modelxyz - np.tile(pointcenter,(modelxyz.shape[0],1))
    lable = np.concatenate([modelxyz, modelvis], 1)
    lable = lable.reshape([1, 6890, 4])

    return samplepointset, lable

def sample_real_pointcloud(pointcloud, num1=20000, num2=10000):
    point_num = pointcloud.shape[0]
    if point_num < max(num1, num2):
        pointcloud = np.tile(pointcloud, [2, 1])
        point_num = pointcloud.shape[0]

    idx = np.random.permutation(point_num)
    idx_num2 = idx[:num2]
    pointcloud_num2 = pointcloud[idx_num2]
    idx_num1 = idx[:num1]
    pointcloud_num1 = pointcloud[idx_num1]

    return pointcloud_num1, pointcloud_num2


def sample_from_depth(image_path, coder=None, num_sample=3000):

    # depthimage = coder.decode_png16(depth_imagedata)
    depthimage = cv2.imread(image_path,-1)

    height, width = depthimage.shape[:2]

    # labelimage = coder.decode_png16(label_imagedata)

    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002
    imx = np.tile(np.arange(width), (height, 1))
    imy = np.tile(np.reshape(np.arange(height), [-1, 1]), (1, width))
    imxlist = np.reshape(imx, [-1])
    imylist = np.reshape(imy, [-1])
    depthlist = np.reshape(np.array(depthimage, dtype=np.float32), [-1]) / 1000.0
    imxlist = (imxlist - cx_d) / fx_d * depthlist
    imylist = (imylist - cy_d) / fy_d * depthlist
    # labellist = np.reshape(np.array(labelimage, dtype=np.int32), [-1]) - 1
    # select = np.reshape(np.where(labellist > -1), [-1])
    # x = np.reshape(imxlist[select], [-1])
    # y = np.reshape(imylist[select], [-1])
    # z = np.reshape(depthlist[select], [-1])
    # vidx = np.reshape(labellist[select], [-1])

    pidx = np.reshape(np.where(depthlist > 0), [-1])
    px = np.reshape(imxlist[pidx], [-1, 1])
    py = np.reshape(imylist[pidx], [-1, 1])
    pz = np.reshape(depthlist[pidx], [-1, 1])
    pointset = np.concatenate([px, py, pz], 1)

    # pointcenter = np.mean(pointset, axis=0)
    #pointset = pointset - np.tile(pointcenter, (pointset.shape[0], 1))

    # modelx = np.zeros(6890)
    # modely = np.zeros(6890)
    # modelz = np.zeros(6890)
    # modelvis = np.zeros(6890)
    # modelx[vidx] = x# - pointcenter[0]  # np.tile(pointcenter[0], (select.shape[0], 1))
    # modely[vidx] = y# - pointcenter[1]  # np.tile(pointcenter[1], (select.shape[0], 1))
    # modelz[vidx] = z# - pointcenter[2]  # np.tile(pointcenter[2], (select.shape[0], 1))
    # nvis = np.array(np.greater(vidx, -1), np.float32)
    # modelvis[vidx] = nvis

    # modelx = np.reshape(modelx, [-1, 1])
    # modely = np.reshape(modely, [-1, 1])
    # modelz = np.reshape(modelz, [-1, 1])
    # modelvis = np.reshape(modelvis, [-1, 1])
    # label = np.concatenate([modelx, modely, modelz, modelvis], 1)
    # modelvt = np.loadtxt(modelfile[frm], dtype=np.float32)
    # label = np.zeros((6890, 4))
    # label[:, 0:3] = modelvt

    # num_sample = 2500  # 6890
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

    # Y1 = samplepointset  # (num_points1, 3)
    # Y2 = modelvt#np.concatenate([modelx, modely, modelz], 1)  # (num_points2, 3)
    # Y1T = np.transpose(Y1, (1, 0))
    # Y2T = np.transpose(Y2, (1, 0))
    # Y3 = np.matmul(np.multiply(Y1, Y1), np.ones(np.shape(Y2T))) + np.matmul(np.ones(np.shape(Y1)),np.multiply(Y2T, Y2T)) - np.multiply(2.0,np.matmul(Y1,Y2T))
    # distance = np.sqrt(Y3)  # (num_points1, num_points2)
    # neg_adj = -distance
    # nn_idx = np.argmax(neg_adj, axis=1)

    # pointclass = smplvertclass[nn_idx]
    # pointclass = np.reshape(pointclass, [num_sample, 1])

    # samplepointset = np.concatenate([samplepointset, pointclass], 1)
    # Samplepointset = np.zeros((num_sample, 4))
    # Samplepointset[:, 0:3] = samplepointset
    # samplepointset_seq[frm] = Samplepointset
    # label_seq[frm] = label
    # frm += 1

    # example = convert_to_example_our_3dpoint_latent_decoder_para_seq(samplepointset_seq, label_seq, label3d)
    # writer.write(example.SerializeToString())
    return pointset # (6890, 3)


if __name__ == "__main__":
    import natsort
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    depth_path = "../DFAUS/CAPE/cape_release/female/depth/smpl_00134_longlong_athletics_trial2_000057_315_0.png"
    vert = sample_from_depth(depth_path)
    vert_20000, _ = sample_real_pointcloud(vert)
    np.savetxt('../fittingcode_allgirl/cape/smpl_00134_longlong_athletics_trial2_000057_315_0_pc.txt', vert_20000)
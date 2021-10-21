""" Evaluates a trained model using placeholders. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import exists

from tf_smpl import projection as proj_util
from tf_smpl.batch_smpl_our import SMPL
from models import get_encoder_fn_separate
from tf_smpl.batch_lbs import batch_rodrigues

para = tf.placeholder(tf.float32, shape=[1,94])
rootR = tf.Variable(tf.zeros(shape=[1,3]), name='globalR')
newcamt = tf.Variable(tf.zeros(shape=[1,3]), name='newcamt')
pose_woR = tf.zeros(shape=[1,69], name='pose_woR')
shape = tf.zeros(shape=[1,10], name='shape')
camR = tf.zeros(shape=[1,9], name='camR')
camt = tf.zeros(shape=[1,3], name='camt')
pose_woR = tf.slice(para,[0,3],[1,69])
shape = tf.slice(para,[0,72],[1,10])
camR = tf.slice(para,[0,82],[1,9])
camR = tf.reshape(camR,[3,3])
camt = tf.slice(para,[0,91],[1,3])

# pose_woR = tf.Variable(para[0,3:72], name='pose_woR')
# shape = tf.Variable(para[0,72:82], name='shape')
# camR = tf.Variable(para[0,82:91], name='camR')
# camt = tf.Variable(para[0,91:94], name='camt')
# camR  = tf.reshape(camR,[3,3])
# pose_woR = tf.reshape(pose_woR, [1,69])
# shape = tf.reshape(shape, [1,10])
# camt = tf.reshape(camt, [1,3])

newpose = tf.concat([rootR,pose_woR],1)

pose = tf.zeros(shape=[1,72], name='pose')
pose = tf.slice(para,[0,0],[1,72])

smpl = SMPL('D:/hmr-master/src/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
verts, pred_Rs = smpl(shape, pose)
verts = tf.squeeze(verts, 0)
verts_rot = tf.matmul(verts, camR)
refverts = verts_rot + tf.tile(tf.reshape(camt, [1, 3]), (6890, 1))

newverts, pred_Rs = smpl(shape, newpose)
newverts = tf.squeeze(newverts, 0)
newverts = newverts + tf.tile(tf.reshape(newcamt, [1, 3]), (6890, 1))

dif = newverts-refverts
loss = tf.reduce_sum(tf.square(dif))#tf.reduce_sum(tf.multiply(dif,dif),[0,1])

opt = tf.train.AdamOptimizer(0.05).minimize(loss)

pname = 'D:/smpl_traindata0/rendermale/allparameters_T5000.txt'
f = open(pname, "w+")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    testpara = np.loadtxt('D:/smpl_traindata0/rendermale/allparameters_RT.txt', dtype=np.float32)
    testpara = np.reshape(testpara,[-1,94])
    samplenum = 5000
    while(samplenum<testpara.shape[0]):#testpara.shape[0]

        tag = 1
        iter = 0
        while(tag):
            _, loss_c, npose, nt, tnewverts, trefverts = sess.run([opt, loss, newpose, newcamt, newverts, refverts], feed_dict={para: np.reshape(testpara[samplenum,:],[1,94])})
            print("saampleidx: %g,iter: %g, loss: %.4f"% (samplenum, iter,loss_c))
            iter = iter+1
            if loss_c<1e-5:
                tag = 0

        newpara = np.concatenate([npose,np.reshape(testpara[samplenum,72:82],[1,10]),nt],1)#
        for i in range(85):
            if i==84:
                f.write(str(newpara[0,i]))
            else:
                f.write(str(newpara[0,i]) + " ")
        f.write("\n")
        samplenum = samplenum+1
    f.close()

    # pname = './tnewverts.txt'  # ''C:/Users/kang/Downloads/hmr-master/tmodel.obj'
    # f = open(pname, "w+")
    # for i in range(6890):
    #     f.write(str(tnewverts[i,0]) + " " + str(tnewverts[i, 1]) + " " + str(tnewverts[i, 2]))
    #     f.write("\n")
    # f.close()
    # pname = './trefverts.txt'  # ''C:/Users/kang/Downloads/hmr-master/tmodel.obj'
    # f = open(pname, "w+")
    # for i in range(6890):
    #     f.write(str(trefverts[i, 0]) + " " + str(trefverts[i, 1]) + " " + str(trefverts[i, 2]))
    #     f.write("\n")
    # f.close()


    # ii = 0
    # vt, Rs, rot= sess.run([finalverts, cams_Rs, cams_rot])
    #
    # v_triangle = np.loadtxt('D:/hmr-master/tri.txt', dtype=np.int32)
    # v_trianglesize = np.shape(v_triangle)[0]
    #
    # #vt = np.squeeze(vt, 0)#self.smpl.verts
    # pname = './smpltest2.obj'#''C:/Users/kang/Downloads/hmr-master/tmodel.obj'
    # f = open(pname, "w+")
    #
    # for i in range(np.shape(vt)[0]):
    #     f.write("v " + str(vt[i,0])+ " "+ str(vt[i,1])+ " "+ str(vt[i,2]))
    #     f.write("\n")
    #
    # for i in range(v_trianglesize):
    #     f.write("f " + str(v_triangle[i,0])+ " "+ str(v_triangle[i,1])+ " "+ str(v_triangle[i,2]))
    #     f.write("\n")
    # f.close()

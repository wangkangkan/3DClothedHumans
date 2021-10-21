"""
    FlowNet3D model with up convolution
"""

import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from utils.pointnet_util import *

from lib.decodeModel import DecodeModel, read_sp_matrix


def get_model(point_cloud, is_training=True, bn_decay=None):

    batch_size = point_cloud['pointcloud'].get_shape()[0].value
    num_point = point_cloud['pointcloud'].get_shape()[1].value - 20000
    l0_xyz_f2 = point_cloud['pointcloud'][:, num_point:, 0:3]

    joint_point = point_cloud["joint_point"]

    decode_model = get_class()

    all_data = {}
    all_data['frm2'] = l0_xyz_f2 

    ############################
    l1_points_f12, _ = extract_jointsfeature(l0_xyz_f2, None, joint_point, is_training=is_training, bn_decay=bn_decay)


    l1_points_f12 = tf.reshape(l1_points_f12, [batch_size, -1])
    net = decode_model._decode_full_4layers(l1_points_f12, use_res_block=True, is_training=is_training)

    return net, all_data

def get_class():

    '''read matrix '''
    A = read_sp_matrix('A')
    A = list(map(lambda x: x.astype('float32'), A))  # float64 -> float32
    U = read_sp_matrix('U')
    U = list(map(lambda x: x.astype('float32'), U))  # float64 -> float32
    D = read_sp_matrix('D')
    D = list(map(lambda x: x.astype('float32'), D))  # float64 -> float32
    L = read_sp_matrix('L')

    p = list(map(lambda x: x.shape[0], A)) 

    params = dict()
    # Architecture.
    nz = 256  # 512
    params['F_0'] = 3  # Number of graph input features.
    params['F'] = [8, 16, 32, 32]  # Number of graph convolutional filters.
    params['K'] = [2] * 4  # Polynomial orders.

    decodermodel = DecodeModel(L, D, U, params['F'], params['K'], p, nz, F_0=params['F_0'])

    return decodermodel

def extract_jointsfeature(point_cloud, point_feat, joints, is_training, bn_decay=None, name="extractglobalfeature"):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    l0_xyz = point_cloud
    l0_points = point_feat

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        l1_points, l1_indices = pointnet_joints_module(l0_xyz, l0_points, joints, radius=0.1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer0', use_nchw=True)

        variables = tf.contrib.framework.get_variables(scope)

        return l1_points, variables

def Laplacian_loss(kp_loader, pred, con):
    grouped_kp_loader = tf.transpose(kp_loader, [1, 0, 2])
    grouped_kp_loader = tf.gather(grouped_kp_loader, con)
    grouped_kp_loader = tf.transpose(grouped_kp_loader, [2, 0, 1, 3])
    # print(grouped_kp_loader)
    grouped_pred = tf.transpose(pred, [1, 0, 2])
    grouped_pred = tf.gather(grouped_pred, con)
    grouped_pred = tf.transpose(grouped_pred, [2, 0, 1, 3])
    # print(grouped_pred)
    # kp_loader = tf.reshape(kp_loader, (-1, 3))
    # pred = tf.reshape(pred, (-1, 3))
    td1 = kp_loader-pred
    res1 = tf.multiply(td1,td1)
    # print(res1)
    td2 = grouped_kp_loader-grouped_pred
    res2 = tf.reduce_mean(tf.multiply(td2,td2),axis=2)
    # print(res2)
    td = res1-res2
    res = tf.reduce_sum(tf.multiply(td,td))
    print(res)
    # exit()
    return res

def edge_loss(pred, gt, vpe):
    '''
    Calculate edge loss measured by difference in the length
    args:
        pred: prediction, [batch size, num_verts (6890), 3]
        gt: ground truth, [batch size, num_verts (6890), 3]
        vpe: SMPL vertex-edges correspondence table, [20664, 2]
    returns:
        edge_obj, an array of size [batch_size, 20664], each element of the second dimension is the
        length of the difference vector between corresponding edges in GT and in pred
    '''
    # get vectors of all edges, have size (batch_size, 20664, 3)
    edges_vec = lambda x: tf.gather(x,vpe[:,0],axis=1) -  tf.gather(x,vpe[:,1],axis=1)
    edge_diff = edges_vec(pred) -edges_vec(gt) # elwise diff between the set of edges in the gt and set of edges in pred
    edge_obj = tf.norm(edge_diff, ord='euclidean', axis=-1)

    return tf.reduce_mean(edge_obj)


def nearestpoint_distance_and_normal(points1, point2, k, pointsnormal1, pointnormal2):
    
    Y1 = points1#(batch_size, num_points1, 3)
    Y2 = point2#(batch_size, num_points2, 3)
    Y1T = tf.transpose(Y1, perm=[0, 2, 1])
    Y2T = tf.transpose(Y2, perm=[0, 2, 1])
    Y3 = tf.matmul(tf.multiply(Y1, Y1), tf.ones(tf.shape(Y2T))) + tf.matmul(tf.ones(tf.shape(Y1)),tf.multiply(Y2T, Y2T)) - tf.multiply(2.0, tf.matmul(Y1, Y2T))
    distance = tf.sqrt(Y3, name='match_relation_matrix')#(batch_size, num_points1, num_points2)
    dot_product = tf.matmul(pointsnormal1,pointnormal2,transpose_b=True)
    # index_mask = tf.where(dot_product>0, )
    # Nmatmul = tf.maximum(dot_product,0)
    Nmatmul = dot_product
    neg_adj = tf.maximum(1-distance/0.1,0)
    temp = tf.multiply(Nmatmul,neg_adj)

    topk, nn_idx = tf.nn.top_k(temp, k=k)
    ones = tf.ones_like(topk)
    zeros = tf.zeros_like(topk)
    mask = tf.where(topk>0,ones,zeros)
    # distance_with_normal = tf.reduce_mean(tf.reduce_sum(topk, axis=[1, 2]))
    return nn_idx, mask

# from evaluatetest import POINT_NUMBER
def normal_loss(points1, point2, pointsnormal1, pointsnormal2, k=1):
    nn_idx, mask = nearestpoint_distance_and_normal(points1, point2, k, pointsnormal1, pointsnormal2)
    # print(nn_idx)
    # print(mask)
    # exit()
    batch_size = point2.get_shape().as_list()[0]
    num_point = point2.get_shape().as_list()[1]
    POINT_NUMBER = points1.get_shape().as_list()[1]

    idx_ = tf.range(batch_size) * num_point
    idx_ = tf.reshape(idx_, [batch_size, 1])
    nn_idx = tf.reshape(nn_idx,[batch_size,-1])
    IDX= nn_idx+idx_
    IDX = tf.reshape(IDX,[-1,1])
    predictvertsnew = tf.reshape(point2,[batch_size*num_point,-1])
    predictvertsnew = tf.gather(predictvertsnew,IDX)
    predictvertsnew = tf.reshape(predictvertsnew,[batch_size,POINT_NUMBER,1,-1])
    gather_pre_vert = tf.squeeze(predictvertsnew,axis=2)
    # gather_pre_vert = tf.gather_nd(point2, nn_idx)

    # loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.reduce_sum((points1 - gather_pre_vert)**2, axis=[2])), axis=1))
    # loss = tf.reduce_sum(mask * tf.reduce_sum(tf.abs(points1 - gather_pre_vert), axis=[2]))

    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(points1 - gather_pre_vert), axis=[1,2]))
    
    # distance2 = nearestpoint_distance_and_normal(point2, points1, k, pointsnormal2, pointsnormal1)

    return loss, gather_pre_vert, nn_idx

def normal_loss_without_normal(points1, point2, idx):
    # nn_idx = nearestpoint_distance_and_normal(points1, point2, k, pointsnormal1, pointsnormal2)
    # print(nn_idx)
    # print(point2)
    # exit()
    idx = tf.cast(idx, dtype=tf.int32)
    batch_size = point2.get_shape().as_list()[0]
    num_point = point2.get_shape().as_list()[1]
    POINT_NUMBER = points1.get_shape().as_list()[1]

    idx_ = tf.range(batch_size) * num_point
    idx_ = tf.reshape(idx_, [batch_size, 1])
    nn_idx = tf.reshape(idx,[batch_size,-1])
    IDX= nn_idx+idx_
    IDX = tf.reshape(IDX,[-1,1])
    predictvertsnew = tf.reshape(point2,[batch_size*num_point,-1])
    predictvertsnew = tf.gather(predictvertsnew,IDX)
    predictvertsnew = tf.reshape(predictvertsnew,[batch_size,POINT_NUMBER,1,-1])
    gather_pre_vert = tf.squeeze(predictvertsnew,axis=2)
    # gather_pre_vert = tf.gather_nd(point2, nn_idx)

    # loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.reduce_sum((points1 - gather_pre_vert)**2, axis=[2])), axis=1))
    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(points1 - gather_pre_vert), axis=[1,2]))
    
    # distance2 = nearestpoint_distance_and_normal(point2, points1, k, pointsnormal2, pointsnormal1)

    return loss, gather_pre_vert, nn_idx

###### model -> pointcloud ############################
def nearestpoint_distance_and_normal_model(points1, point2, k, pointsnormal1, pointnormal2):
    
    Y1 = points1#(batch_size, num_points1, 3)
    Y2 = point2#(batch_size, num_points2, 3)
    Y1T = tf.transpose(Y1, perm=[0, 2, 1])
    Y2T = tf.transpose(Y2, perm=[0, 2, 1])
    Y3 = tf.matmul(tf.multiply(Y1, Y1), tf.ones(tf.shape(Y2T))) + tf.matmul(tf.ones(tf.shape(Y1)),tf.multiply(Y2T, Y2T)) - tf.multiply(2.0, tf.matmul(Y1, Y2T))
    distance = tf.sqrt(Y3, name='match_relation_matrix')#(batch_size, num_points1, num_points2)
    dot_product = tf.matmul(pointsnormal1,pointnormal2,transpose_b=True)
    # index_mask = tf.where(dot_product>0, )
    Nmatmul = tf.maximum(dot_product,0)
    neg_adj = tf.maximum(1-distance/0.3,0)
    temp = tf.multiply(Nmatmul,neg_adj)

    topk, nn_idx = tf.nn.top_k(temp, k=k)
    temp_for_mask = tf.reduce_sum(dot_product, axis=-1)
    ones = tf.ones_like(temp_for_mask)
    zeros = tf.zeros_like(temp_for_mask)
    mask = tf.where(temp_for_mask>0,ones,zeros)
    # distance_with_normal = tf.reduce_mean(tf.reduce_sum(topk, axis=[1, 2]))
    return nn_idx, mask

# from evaluatetest import POINT_NUMBER
def normal_loss_model(points1, point2, pointsnormal1, pointsnormal2, k=1):
    nn_idx, mask = nearestpoint_distance_and_normal_model(points1, point2, k, pointsnormal1, pointsnormal2)
    # print(nn_idx)
    # print(point2)
    # exit()
    batch_size = point2.get_shape().as_list()[0]
    num_point = point2.get_shape().as_list()[1]
    POINT_NUMBER = points1.get_shape().as_list()[1]

    idx_ = tf.range(batch_size) * num_point
    idx_ = tf.reshape(idx_, [batch_size, 1])
    nn_idx = tf.reshape(nn_idx,[batch_size,-1])
    IDX= nn_idx+idx_
    IDX = tf.reshape(IDX,[-1,1])
    predictvertsnew = tf.reshape(point2,[batch_size*num_point,-1])
    predictvertsnew = tf.gather(predictvertsnew,IDX)
    predictvertsnew = tf.reshape(predictvertsnew,[batch_size,POINT_NUMBER,1,-1])
    gather_pre_vert = tf.squeeze(predictvertsnew,axis=2)
    # gather_pre_vert = tf.gather_nd(point2, nn_idx)

    # loss = tf.reduce_mean(tf.reduce_sum(mask * tf.sqrt(tf.reduce_sum((points1 - gather_pre_vert)**2, axis=[2])), axis=1))
    loss = tf.reduce_mean(tf.reduce_sum(mask * tf.reduce_sum(tf.abs(points1 - gather_pre_vert), axis=[2]), axis=[1]))
    
    # distance2 = nearestpoint_distance_and_normal(point2, points1, k, pointsnormal2, pointsnormal1)

    return loss, gather_pre_vert

def cal_normals(vert, face):
    batch_size, point_num, channel = vert.get_shape().as_list()
    new_vert_0 = tf.gather(vert, indices=face[:, 0], axis=1)
    new_vert_1 = tf.gather(vert, indices=face[:, 1], axis=1)
    new_vert_2 = tf.gather(vert, indices=face[:, 2], axis=1)

    normal = tf.cross(new_vert_1 - new_vert_0, new_vert_2 - new_vert_0, name="normals")

    normal = tf.nn.l2_normalize(tf.reduce_mean(tf.reshape(normal, [batch_size, point_num, -1, channel]), axis=2), axis=-1)
    return normal

def mesh_4x(edge, vert):
    new_vert_0 = tf.gather(vert, indices=edge[:, 0], axis=1)
    new_vert_1 = tf.gather(vert, indices=edge[:, 1], axis=1)
    new_vert = (new_vert_0 + new_vert_1) / 2
    return tf.concat([vert, new_vert], axis=1, name="mesh_4x_concat")

def get_loss(pred, label):
    """ pred: BxNx3,
        label: BxNx3,

    """

    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value
    channel = label.get_shape()[2].value

    l2_loss = tf.reduce_sum(tf.reduce_sum(tf.abs(pred-label), axis=2)) / batch_size

    return l2_loss
def get_loss_display(pred, label):
    """ pred: BxNx3,
        label: BxNx3,
        mask: BxN
    """


    batch_size = pred.get_shape()[0].value

    l2_loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum((pred-label)**2, axis=[2]))) / batch_size

    return l2_loss


if __name__=='__main__':

    from write2obj import read_obj, np
    pred_verts = tf.placeholder(tf.float32, shape=[1, 16535, 3])
    faces_tf = tf.placeholder(tf.int32, shape=[None, 3])

    meshnormal = cal_normals(pred_verts, faces_tf)

    with tf.Session() as sess:
        vert, face = read_obj('./test/new_face.obj')
        face = np.loadtxt('./normal_face_x6.txt', dtype=np.int) - 1
        for i in range(1):
            result = sess.run(meshnormal, feed_dict={pred_verts:vert[np.newaxis, :, :], faces_tf:face})
        print(result.shape)
        np.savetxt('test_normal.txt', np.squeeze(result, axis=0))
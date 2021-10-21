""" PointNet++ Layers

Original Author: Charles R. Qi
Modified by Xingyu Liu
Date: April 2019
"""

import os
import sys

import pointnet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def down_sample_fun(npoint, radius, nsample, xyz, points, index, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    # new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    # new_xyz = xyz[:, index, :]
    index = tf.convert_to_tensor(index, dtype=tf.int32)
    new_xyz = tf.gather(xyz, indices=index, axis=1)

    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_with_joints_and_group(joints, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    # print('xyz:',xyz.shape)
    # tempxyz = farthest_point_sample(npoint, xyz)
    # print('tempxyz',tempxyz.shape)

    # new_xyz = gather_point(xyz, tempxyz) # (batch_size, npoint, 3)
    # print('new_xyz',new_xyz.shape)
    # sidx = np.array([1,50,101,201,301,401,501,601,701,801.901,1001,1101,1201,1301,1401,1501,1601,1701,1801,1901,2001,2101,2201,2295])
    # sidx = np.tile(sidx,[16,1]) 
    # # print('sidx',sidx.shape)
    # joints = gather_point(xyz,sidx)
    # batch_size = xyz.get_shape()[0].value
    #joints = tf.reshape(joints,[1,24,-1])
    #joints = tf.tile(joints, multiples=[batch_size, 1, 1])
    # print(joints.shape)
    if knn:
        _,idx = knn_point(nsample, xyz, joints)
        # print('Knn idx:',idx.shape)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, joints)
        label = tf.cast(tf.greater(pts_cnt, 0),dtype = tf.int32)
        labelexpand = tf.tile(tf.expand_dims(label,-1),[1,1,nsample])
        # labelcast = tf.cast(labelexpand,dtype = tf.int32)
        idx = tf.multiply(idx,labelexpand)
        # print('idx:',idx.shape)
        # print('pts_snt:',pts_cnt.shape)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    # print('grouped_xyz:',grouped_xyz.shape)
    grouped_xyz -= tf.tile(tf.expand_dims(joints, 2), [1,1,nsample,1]) # translation normalization
    # print('jointsex:',tf.expand_dims(joints, 2).shape)
    # print('grouped_xyz:',grouped_xyz.shape)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        # print('grouped_points:',grouped_points.shape)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            # print('new_points1:',new_points.shape)
        else:
            new_points = grouped_points
            # print('new_points2:',new_points.shape)
    else:
        new_points = grouped_xyz
        # print('new_points3:',new_points.shape)
    # with self.test_session():
    #     print("---- Going to compute gradient error")
    #     err = tf.test.compute_gradient_error(points, (1,128,16), grouped_points, (1,8,32,16))
    #     print(err)
    #     self.assertLess(err, 1e-4) 
    # print("---- Going to compute gradient error")
    # err = tf.test.compute_gradient_error(joints, (16,24,3), grouped_xyz, (16,24,64,3))
    # print(err)
    # self.assertLess(err, 1e-4) 
    return new_points, idx, grouped_xyz, label



def down_sample_fun_simple(xyz, index):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    # new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    # new_xyz = xyz[:, index, :]
    index = tf.convert_to_tensor(index, dtype=tf.int32)
    new_xyz = tf.gather(xyz, indices=index, axis=1)

    # if knn:
    #     _,idx = knn_point(nsample, xyz, new_xyz)
    # else:
    #     idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    # grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    # grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    # if points is not None:
    #     grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
    #     if use_xyz:
    #         new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
    #     else:
    #         new_points = grouped_points
    # else:
    #     new_points = grouped_xyz

    return new_xyz#, new_points, idx, grouped_xyz


def sample_and_group_layer1(npoint, classes, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    # print(classes)
    xyz = tf.concat([xyz, classes], axis=-1)
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    print(new_xyz)

    new_xyz = new_xyz[:, :, :3]
    new_class = new_xyz[:, :, 3:]
    exit()
    xyz = xyz[:, :, :3]
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz, tf.reshape(new_class, [-1, npoint, 1])

def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz

def pointnet_sa_module_layer1(xyz, points, npoint, classes, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz, new_class = sample_and_group_layer1(npoint, classes, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
                                        # data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay)
                                            # data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx, new_class


############# MLP #############################################################
def MLP_2d(features, mlp, is_training, bn_decay, bn, scope='mlp2d', padding='VALID'):
    for i, num_out_channel in enumerate(mlp):
        features = tf_util.conv2d(features, num_out_channel, [1,1],
                                            padding=padding, stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope= scope+'_%d'%(i), bn_decay=bn_decay) 
    return features


############ coeff ###########################################################
def coeff_generation(grouped_features, features, grouped_xyz, is_training, bn_decay, scope, bn=True, mode='with_feature'):
    gac_par = [
        [32, 16], #MLP for xyz 
        [16, 16], #MLP for feature 
        [64] #hidden node of MLP for mergering 
    ]
    with tf.variable_scope(scope) as sc:
        if mode == 'with_feature':
            coeff = grouped_features - features
            coeff = MLP_2d(coeff, gac_par[1], is_training, bn_decay, bn, scope='conv_with_feature')## compress to a hiden feature space  
            coeff = tf.concat([grouped_xyz, coeff], -1)
        # print(coeff)
        # out_chal = grouped_features.get_shape()[-1].value
        grouped_features = tf.concat([grouped_xyz, grouped_features], axis=-1) #updata feature

        out_chal = grouped_features.get_shape()[-1].value
        coeff = MLP_2d(coeff, gac_par[2], is_training, bn_decay, bn, scope='conv')## map to a hiden feature space     
        coeff = tf_util.conv2d(coeff, out_chal, [1,1], scope='conv2d', is_training=is_training, bn_decay=bn_decay, activation_fn=None)  #output coefficent
        coeff = tf.nn.softmax(coeff, dim=2) #coefffient normalization

        grouped_features = tf.multiply(coeff, grouped_features)
        # print(grouped_features)
        grouped_features = tf.reduce_max(grouped_features, axis=[2], keep_dims=False, name='maxpool')#tf.reduce_sum(grouped_features, axis=[2], keep_dims=False)
        # grouped_features = tf.reduce_sum(grouped_features, axis=[2], keep_dims=False)

        return grouped_features

################## chebyshev5 #######################################################
def chebyshev5(grouped_feature, x, grouped_xyz, bn_decay=None):
    # grouped_feature = tf.transpose(x, [1, 0, 2])
    # grouped_feature = tf.gather(grouped_feature, con)
    # grouped_feature = tf.transpose(grouped_feature, [2, 0, 1, 3])

    # grouped_xyz = tf.transpose(xyz, [1, 0, 2])
    # grouped_xyz = tf.gather(grouped_xyz, con)
    # grouped_xyz = tf.transpose(grouped_xyz, [2, 0, 1, 3])
    # grouped_xyz -= tf.expand_dims(xyz, 2) # translation normalization
    # print(grouped_feature)
    # print(x)
    # print(grouped_xyz)
    new_feature = coeff_generation(grouped_feature, x, grouped_xyz, tf.cast(True, tf.bool), bn_decay, scope='space_coeff_gen')
    # print(new_feature)
    # exit()
    # new_feature = tf.concat([x, new_feature], axis=-1) 
    # new_feature = self.featuremap(new_feature,  Fout)

    # N, M, Fin = new_feature.get_shape()
    return new_feature#tf.reshape(new_feature, [-1, seq, M, Fin])  # N x M x Fout


def pointnet_joints_module(xyz, points, joints, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_points, idx, grouped_xyz, label = sample_with_joints_and_group(joints, radius, nsample, xyz, points, knn, use_xyz)
        # print(grouped_xyz)
        # print(joints)
        # print(new_points)
        # Point Feature Embedding
        input_image = tf.expand_dims(joints, -1)
        with tf.variable_scope("pointnet") as scope:
            # Point functions (MLP implemented as conv2d)
            net = tf_util.conv2d(input_image, mlp[0], [1,3],
                                padding='VALID', stride=[1,1],
                                bn=True, is_training=is_training,
                                scope='conv1', bn_decay=bn_decay)
            net = tf_util.conv2d(net, mlp[1], [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv2', bn_decay=bn_decay)
            net = tf_util.conv2d(net, mlp[2], [1,1],
                                padding='VALID', stride=[1,1],
                                bn=True, is_training=is_training,
                                scope='conv3', bn_decay=bn_decay)
        # print(net)
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
            # print('new_points:',i,'  ',new_points.shape)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])
        # for i, num_out_channel in enumerate(mlp):
        #     grouped_xyz = tf_util.conv2d(grouped_xyz, num_out_channel, [1,1],
        #                                 padding='VALID', stride=[1,1],
        #                                 bn=bn, is_training=is_training,
        #                                 scope='xyz_conv%d'%(i), bn_decay=bn_decay)
                                        # data_format=data_format) 

        new_feature = chebyshev5(new_points, net, grouped_xyz, bn_decay=None)
        # Pooling in Local Regions
        # if pooling=='max':
        #     new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        # elif pooling=='avg':
        #     new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        # elif pooling=='weighted_avg':
        #     with tf.variable_scope('weighted_avg'):
        #         dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
        #         exp_dists = tf.exp(-dists * 5)
        #         weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
        #         new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
        #         new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        # elif pooling=='max_and_avg':
        #     max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        #     avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        #     new_points = tf.concat([avg_points, max_points], axis=-1)

        # # [Optional] Further Processing 
        # if mlp2 is not None:
            # if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            # for i, num_out_channel in enumerate(mlp2):
            #     new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
            #                                 padding='VALID', stride=[1,1],
            #                                 bn=bn, is_training=is_training,
            #                                 scope='conv_post_%d'%(i), bn_decay=bn_decay,
            #                                 data_format=data_format) 
            # if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])
        # print(new_points)
        # new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        labelexpand = tf.cast(tf.tile(tf.expand_dims(label,-1),[1,1,mlp[-1]+3]),dtype=tf.float32)
        new_points = tf.multiply(new_feature,labelexpand)
        # print(new_points)
        return new_points, idx




def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=False, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    print('sasasasasasasasasasasasasasasasasasasasasasasasasasa')
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
                                        # data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay)
                                            # data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def pointnet_sa_module_concat(xyz1, xyz2,  points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    print('concat'*10)
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        # if group_all:
        #     nsample = xyz.get_shape()[1].value
        #     new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        # else:
        #     new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)
        # new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
   
    
        idx, pts_cnt = query_ball_point(radius, nsample, xyz2, xyz1)

        # _, idx_knn = knn_point(nsample, xyz2, xyz1)
        # cnt = tf.tile(tf.expand_dims(pts_cnt, -1), [1,1,nsample])
        # idx = tf.where(cnt > (nsample-1), idx, idx_knn)
        
        label = tf.cast(tf.greater(pts_cnt, 0),dtype = tf.int32)
        labelcast = tf.cast(tf.tile(tf.expand_dims(label, -1), [1,1,nsample]),dtype = tf.int32)
        idx = tf.multiply(idx,labelcast)
        
        # grouped_xyz = group_point(xyz2, idx) # (batch_size, npoint, nsample, 3)
        # grouped_xyz -= tf.tile(tf.expand_dims(xyz1, 2), [1,1,nsample,1])
        # xyzdiff = grouped_xyz



        labelexpand = tf.tile(tf.expand_dims(tf.expand_dims(label,-1), -1),[1,1,nsample, 3])
        grouped_xyz1 = tf.tile(tf.expand_dims(xyz1, 2), [1, 1, nsample, 1])
        grouped_xyz = group_point(xyz2, idx) # (batch_size, npoint, nsample, 3)
        grouped_xyz = tf.where(labelexpand>0, grouped_xyz, grouped_xyz1)
        xyzdiff = grouped_xyz -  grouped_xyz1 # translation normalization
        if points is not None:
            grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
            if use_xyz:
                new_points = tf.concat([xyzdiff, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            else:
                new_points = grouped_points
        else:
            new_points = tf.concat([grouped_xyz, xyzdiff], axis=3)
            # new_points = xyzdiff


        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
                                        # data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # local feature transform
        for i, num_out_channel in enumerate(mlp):
            local_feat = tf_util.conv1d(xyz1, num_out_channel, 1,
                                        padding='VALID', stride=1,
                                        bn=bn, is_training=is_training,
                                        scope='con_1d_%d'%(i), bn_decay=bn_decay)

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                weights =  tf_util.conv2d(new_points-tf.expand_dims(local_feat, axis=2), mlp[-2], [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_weight_1', bn_decay=bn_decay)
                weights =  tf_util.conv2d(weights, mlp[-1], [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_weight_2', bn_decay=bn_decay)
                weights = tf.nn.softmax(weights, axis=2)
                
                # weights = tf.where(weights > 0.1, x=weights, y=tf.zeros_like(weights))
                # weights = weights / tf.reduce_sum(weights, axis=2, keepdims=True)
                # weights = tf.where(weights > 0.1, x=weights, y=tf.zeros_like(weights))
                # weights = weights / tf.reduce_sum(weights, axis=2, keepdims=True)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)

                num_outputs= new_points.get_shape()[-1].value
                biases = tf_util._variable_on_cpu('biases', [num_outputs], tf.constant_initializer(0.0))
                new_points = tf.nn.bias_add(new_points, biases)
                
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay)
                                            # data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        new_points = tf.concat([new_points, local_feat], axis=-1)
        # new_points = tf.concat([xyz1, new_points], axis=-1)

        return xyz1, new_points, idx



def pointnet_sa_module_down_sample(xyz, points, npoint, radius, nsample, index, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=False, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = down_sample_fun(npoint, radius, nsample, xyz, points, index, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
                                        # data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay)
                                            # data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx



def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True, last_mlp_activation=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp)-1 and not(last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = tf.nn.relu
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay, activation_fn=activation_fn)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

def flow_embedding_module(batch_size, xyz1, xyz2, feat1, feat2, radius, nsample, mlp, is_training, bn_decay, scope, bn=True, pooling='max', knn=True, corr_func='elementwise_product'):
    """
    Input:
        xyz1: (batch_size, npoint, 3)
        xyz2: (batch_size, npoint, 3)
        feat1: (batch_size, npoint, channel)
        feat2: (batch_size, npoint, channel)
    Output:
        xyz1: (batch_size, npoint, 3)
        feat1_new: (batch_size, npoint, mlp[-1])
    """
    # xyz1 = tf.reshape(xyz1, [batch_size, -1, 3])
    # xyz2 = tf.reshape(xyz2, [batch_size, -1, 3])
    # feat1 = tf.reshape(feat1, [batch_size, -1, 3])
    # feat2 = tf.reshape(feat2, [batch_size, -1, 3])
    if knn:
        _, idx = knn_point(nsample, xyz2, xyz1)
    else:
        idx, cnt = query_ball_point(radius, nsample, xyz2, xyz1)
        _, idx_knn = knn_point(nsample, xyz2, xyz1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1,1,nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)

    xyz2_grouped = group_point(xyz2, idx) # batch_size, npoint, nsample, 3
    xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint, 1, 3
    xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint, nsample, 3

    feat2_grouped = group_point(feat2, idx) # batch_size, npoint, nsample, channel
    feat1_expanded = tf.expand_dims(feat1, 2) # batch_size, npoint, 1, channel
    # TODO: change distance function
    if corr_func == 'elementwise_product':
        feat_diff = feat2_grouped * feat1_expanded # batch_size, npoint, nsample, channel
    elif corr_func == 'concat':
        feat_diff = tf.concat(axis=-1, values=[feat2_grouped, tf.tile(feat1_expanded,[1,1,nsample,1])]) # batch_size, npoint, sample, channel*2
    elif corr_func == 'dot_product':
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
    elif corr_func == 'cosine_dist':
        feat2_grouped = tf.nn.l2_normalize(feat2_grouped, -1)
        feat1_expanded = tf.nn.l2_normalize(feat1_expanded, -1)
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
    elif corr_func == 'flownet_like': # assuming square patch size k = 0 as the FlowNet paper
        batch_size = xyz1.get_shape()[0].value
        npoint = xyz1.get_shape()[1].value
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
        total_diff = tf.concat(axis=-1, values=[xyz_diff, feat_diff]) # batch_size, npoint, nsample, 4
        feat1_new = tf.reshape(total_diff, [batch_size, npoint, -1]) # batch_size, npoint, nsample*4
        #feat1_new = tf.concat(axis=[-1], values=[feat1_new, feat1]) # batch_size, npoint, nsample*4+channel
        return xyz1, feat1_new


    feat1_new = tf.concat([feat_diff, xyz_diff], axis=3) # batch_size, npoint, nsample, [channel or 1] + 3
    # TODO: move scope to outer indent
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='conv_diff_%d'%(i), bn_decay=bn_decay)
    if pooling=='max':
        feat1_new = tf.reduce_max(feat1_new, axis=[2], keep_dims=False, name='maxpool_diff')
    elif pooling=='avg':
        feat1_new = tf.reduce_mean(feat1_new, axis=[2], keep_dims=False, name='avgpool_diff')
    return xyz1, feat1_new


def flow_embedding_module_small(batch_size, xyz1, xyz2, xyz4, feat2, radius, nsample, mlp, is_training, bn_decay, scope, bn=True, pooling='max', knn=True, corr_func='elementwise_product'):
    """
    Input:
        xyz1: (batch_size, npoint, 3)
        xyz2: (batch_size, npoint, 3)
        feat1: (batch_size, npoint, channel)
        feat2: (batch_size, npoint, channel)
    Output:
        xyz1: (batch_size, npoint, 3)
        feat1_new: (batch_size, npoint, mlp[-1])
    """
    # xyz1 = tf.reshape(xyz1, [batch_size, -1, 3])
    # xyz2 = tf.reshape(xyz2, [batch_size, -1, 3])
    # feat1 = tf.reshape(feat1, [batch_size, -1, 3])
    # feat2 = tf.reshape(feat2, [batch_size, -1, 3])
    xyz3 = xyz1 + xyz2
    if knn:
        _, idx = knn_point(nsample, xyz3, xyz3)
    else:
        idx, cnt = query_ball_point(radius, nsample, xyz2, xyz1)
        _, idx_knn = knn_point(nsample, xyz2, xyz1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1,1,nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)
    xyz2_grouped = group_point(xyz3, idx) # batch_size, npoint, nsample, 3
    # xyz2_grouped = tf.tile(tf.expand_dims(xyz2, axis=2), [1,1,nsample,1]) # batch_size, npoint, nsample, 3
    # feat2_grouped = tf.tile(tf.expand_dims(feat2, axis=2), [1,1,nsample,1]) # batch_size, npoint, nsample, 3
    # xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint, 1, 3
    # xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint, nsample, 3

    feat2_grouped = group_point(feat2, idx) # batch_size, npoint, nsample, channel
    # feat1_expanded = tf.expand_dims(feat1, 2) # batch_size, npoint, 1, channel
    # TODO: change distance function
    # if corr_func == 'elementwise_product':
    #     feat_diff = feat2_grouped * feat1_expanded # batch_size, npoint, nsample, channel
    # elif corr_func == 'concat':
    #     feat_diff = tf.concat(axis=-1, values=[feat2_grouped, tf.tile(feat1_expanded,[1,1,nsample,1])]) # batch_size, npoint, sample, channel*2
    # elif corr_func == 'dot_product':
    #     feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
    # elif corr_func == 'cosine_dist':
    #     feat2_grouped = tf.nn.l2_normalize(feat2_grouped, -1)
    #     feat1_expanded = tf.nn.l2_normalize(feat1_expanded, -1)
    #     feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
    # elif corr_func == 'flownet_like': # assuming square patch size k = 0 as the FlowNet paper
    #     batch_size = xyz1.get_shape()[0].value
    #     npoint = xyz1.get_shape()[1].value
    #     feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
    #     total_diff = tf.concat(axis=-1, values=[xyz_diff, feat_diff]) # batch_size, npoint, nsample, 4
    #     feat1_new = tf.reshape(total_diff, [batch_size, npoint, -1]) # batch_size, npoint, nsample*4
    #     #feat1_new = tf.concat(axis=[-1], values=[feat1_new, feat1]) # batch_size, npoint, nsample*4+channel
    #     return xyz1, feat1_new


    feat1_new = tf.concat([feat2_grouped, xyz2_grouped], axis=3) # batch_size, npoint, nsample, [channel or 1] + 3
    # TODO: move scope to outer indent
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='conv_diff_%d'%(i), bn_decay=bn_decay)
    if pooling=='max':
        feat1_new = tf.reduce_max(feat1_new, axis=[2], keep_dims=False, name='maxpool_diff')
    elif pooling=='avg':
        feat1_new = tf.reduce_mean(feat1_new, axis=[2], keep_dims=False, name='avgpool_diff')
    return xyz1, feat1_new

def set_upconv_module(xyz1, xyz2, feat1, feat2, nsample, mlp, mlp2, is_training, scope, bn_decay=None, bn=True, pooling='max', radius=None, knn=True):
    """
        Feature propagation from xyz2 (less points) to xyz1 (more points)

    Inputs:
        xyz1: (batch_size, npoint1, 3)
        xyz2: (batch_size, npoint2, 3)
        feat1: (batch_size, npoint1, channel1) features for xyz1 points (earlier layers)
        feat2: (batch_size, npoint2, channel2) features for xyz2 points
    Output:
        feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

        TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
    """
    with tf.variable_scope(scope) as sc:
        if knn:
            l2_dist, idx = knn_point(nsample, xyz2, xyz1)
        else:
            idx, pts_cnt = query_ball_point(radius, nsample, xyz2, xyz1)
        xyz2_grouped = group_point(xyz2, idx) # batch_size, npoint1, nsample, 3
        xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint1, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint1, nsample, 3

        feat2_grouped = group_point(feat2, idx) # batch_size, npoint1, nsample, channel2
        net = tf.concat([feat2_grouped, xyz_diff], axis=3) # batch_size, npoint1, nsample, channel2+3

        if mlp is None: mlp=[]
        for i, num_out_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_out_channel, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv%d'%(i), bn_decay=bn_decay)
        if pooling=='max':
            feat1_new = tf.reduce_max(net, axis=[2], keep_dims=False, name='maxpool') # batch_size, npoint1, mlp[-1]
        elif pooling=='avg':
            feat1_new = tf.reduce_mean(net, axis=[2], keep_dims=False, name='avgpool') # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = tf.concat([feat1_new, feat1], axis=2) # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = tf.expand_dims(feat1_new, 2) # batch_size, npoint1, 1, mlp[-1]+channel2
        if mlp2 is None: mlp2=[]
        for i, num_out_channel in enumerate(mlp2):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='post-conv%d'%(i), bn_decay=bn_decay)
        feat1_new = tf.squeeze(feat1_new, [2]) # batch_size, npoint1, mlp2[-1]
        return feat1_new

def pointclould_classfy(xyz, classnum, pointfeature, classtensor, classeslabel, is_training):
    # batch_size = xyz.get_shape()[0].value
    # num_point = xyz.get_shape()[1].value
    # pointclass, variables = pointnet.get_model_class(xyz, classnum, is_training)
    # pointclass_index = tf.argmax(pointclass, axis=-1) # BxN
    # # print('pointclass_index:',pointclass_index)

    # pointclass_list = []
    # featureclass_list = []
    # for batch in range(batch_size):
    #     # featureclass_list = []
    #     for i in range(classnum):
    #         equal_class = tf.equal(pointclass_index[batch], tf.zeros([num_point], dtype=tf.int64) + i+1)
    #         class_index = tf.where(equal_class)
    #         # print(tf.reduce_sum(tf.cast(equal_class, dtype=tf.int32)))
    #         class_index = tf.squeeze(class_index, axis=1)
    #         # class_index = tf.reshape(class_index, [batch_size, -1])
    #         pointclass_i = tf.gather(xyz[batch], indices=class_index, axis=0)
    #         # print('equal_class', pointclass_i)
    #         # exit()
    #         featureclass_i = tf.gather(feature[batch], indices=class_index, axis=0)
    #         pointclass_list.append(pointclass_i)
    #         featureclass_list.append(featureclass_i)
    #     # featureclass_list.append(featureclass_i)


    # return pointclass_list, featureclass_list, pointclass, variables
    patch_feat, patch_point, pointclass, variables = pointnet.get_model_withoutclass(xyz, pointfeature, classnum, classtensor, classeslabel, is_training)
    return patch_point, patch_feat, pointclass, variables

def makesure_dim_equal(xyz1, xyz2, feature1, feature2, batch_size, classnum):
    # for i in range(len(xyz1)):
    #     item1 = xyz1[i]
    #     item11 = feature1[i]
    #     item2 = xyz2[i]
    #     item22 = feature2[i]
    #     num_point1 = item1.get_shape()[0].value
    #     num_point2 = item2.get_shape()[0].value
    #     print(num_point1)
    #     if num_point1 < num_point2:
    #         temp_point = item2[:num_point1, :]
    #         temp_feat = item22[:num_point1, :]
    #         xyz2[i] = temp_point
    #         feature2[i] = temp_feat
    #     elif num_point1 > num_point2:
    #         repeat_point = tf.tile(tf.expand_dims(item2[-1, :], axis=0), [num_point1-num_point2, 1])
    #         temp_point = tf.concat([item2, repeat_point], axis=0)
    #         repeat_feat = tf.tile(tf.expand_dims(item22[-1, :], axis=0), [num_point1-num_point2, 1])
    #         temp_feat = tf.concat([item22, repeat_feat], axis=0)
    #         xyz2[i] = temp_point
    #         feature2[i] = temp_feat
    xyzpoint1, xyzpoint2, xyzfeat1, xyzfeat2 = [], [], [], []
    for i in range(classnum):
        temp_point_1 = xyz1[i::classnum]
        temp_point_2 = xyz2[i::classnum]
        temp_feat_1 = feature1[i::classnum]
        temp_feat_2 = feature2[i::classnum]
        xyzpoint1.append(tf.concat(temp_point_1, axis=0))
        xyzpoint2.append(tf.concat(temp_point_2, axis=0))
        xyzfeat1.append(tf.concat(temp_feat_1, axis=0))
        xyzfeat2.append(tf.concat(temp_feat_2, axis=0))
    return xyzpoint1, xyzpoint2, xyzfeat1, xyzfeat2

def choose_class(classes, xyz, npoint):
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
    # idx, pts_cnt = knn_point(1, xyz, new_xyz)
    idx, pts_cnt = query_ball_point(1e-19, 1, xyz, new_xyz)
    grouped_class = group_point(classes, idx)
    grouped_class = tf.squeeze(grouped_class, axis=2)
    return grouped_class
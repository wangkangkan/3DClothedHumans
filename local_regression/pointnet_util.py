""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys

ROOT_DIR = "./"  # important
# exit()

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
    print('xyz:',xyz.shape)
    tempxyz = farthest_point_sample(npoint, xyz)
    print('tempxyz',tempxyz.shape)

    new_xyz = gather_point(xyz, tempxyz) # (batch_size, npoint, 3)
    print('new_xyz',new_xyz.shape)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
        print('idx:',idx.shape)
        print('pts_snt:',pts_cnt.shape)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    print('grouped_xyz:',grouped_xyz.shape)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    print('new_xyzex:',tf.expand_dims(new_xyz, 2).shape)
    print('grouped_xyz:',grouped_xyz.shape)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        print('grouped_points:',grouped_points.shape)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            print('new_points1:',new_points.shape)
        else:
            new_points = grouped_points
            print('new_points2:',new_points.shape)
    else:
        new_points = grouped_xyz
        print('new_points3:',new_points.shape)
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
from models import *
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
        coeff = conv2d(coeff, out_chal, [1,1], scope='conv2d', is_training=is_training, bn_decay=bn_decay, activation_fn=None)  #output coefficent
        coeff = tf.nn.softmax(coeff, dim=2) #coefffient normalization

        grouped_features = tf.multiply(coeff, grouped_features)
        # print(grouped_features)
        grouped_features = tf.reduce_max(grouped_features, axis=[2], keep_dims=False, name='maxpool')#tf.reduce_sum(grouped_features, axis=[2], keep_dims=False)
        # grouped_features = tf.reduce_sum(grouped_features, axis=[2], keep_dims=False)

        return grouped_features


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
            net = tf_util.conv2d(input_image, 64, [1,3],
                                padding='VALID', stride=[1,1],
                                bn=True, is_training=is_training,
                                scope='conv1', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 64, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv2', bn_decay=bn_decay)
            net = tf_util.conv2d(net, 128, [1,1],
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

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
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
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
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
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
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

 
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
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
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1
import tensorflow as tf
import tf_util
from pointnet_util import pointnet_sa_module

def pointnet_feature(point_cloud, is_training, bn_decay=None):

    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = None

    with tf.variable_scope("pointnet") as scope:

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        variables = tf.contrib.framework.get_variables(scope)

        return net, variables

def extractfeature_onrawpointsets(point_cloud, npoint, radius, nsample, mlp, scopename, is_training, reuse, bn_decay=None):

    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = None

    with tf.variable_scope(scopename, reuse=reuse) as scope:

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint, radius, nsample, mlp, mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)

        variables = tf.contrib.framework.get_variables(scope)

        return l1_xyz, l1_points, variables

def extractfeature_onallrawpointsets(point_cloud, tmlp, scopename, is_training, reuse, bn_decay=None):

    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = None

    with tf.variable_scope(scopename, reuse=reuse) as scope:

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        #l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint, radius, nsample, mlp, mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=None, radius=None, nsample=None,mlp=tmlp, mlp2=None, group_all=True,is_training=is_training, bn_decay=bn_decay, scope='layer1')
        variables = tf.contrib.framework.get_variables(scope)

        return l1_xyz, l1_points, variables

def parameter_regression(point_cloud, point_feat, outdim, is_training, bn_decay=None):

    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = point_feat

    with tf.variable_scope("regression") as scope:

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
        net = tf_util.fully_connected(net, outdim, activation_fn=None, scope='fc3')

        variables = tf.contrib.framework.get_variables(scope)

        return net, variables

def extract_globalfeature(point_cloud, point_feat, is_training, bn_decay=None):

    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = point_feat

    with tf.variable_scope("extractglobalfeature") as scope:

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])

        variables = tf.contrib.framework.get_variables(scope)

        return net, variables

def extract_globalfeature_twolayer(point_cloud, point_feat, is_training, bn_decay=None):

    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = point_feat

    with tf.variable_scope("extractglobalfeature_twolayer") as scope:

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.

        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer2')

        # Fully connected layers
        net = tf.reshape(l2_points, [batch_size, -1])

        variables = tf.contrib.framework.get_variables(scope)

        return net, variables
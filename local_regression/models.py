"""
Defines networks.

@Encoder_resnet
@Encoder_resnet_v1_101
@Encoder_fc3_dropout

@Discriminator_separable_rotations

Helper:
@get_encoder_fn_separate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer


def Encoder_resnet(x, is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    with tf.name_scope("Encoder_resnet", [x]):
        with slim.arg_scope(
                resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            net, end_points = resnet_v2.resnet_v2_50(
                x,
                num_classes=None,
                is_training=is_training,
                reuse=reuse,
                scope='resnet_v2_50')
            net = tf.squeeze(net, axis=[1, 2])
    variables = tf.contrib.framework.get_variables('resnet_v2_50')
    return net, variables


def Encoder_fc3_dropout(x,
                        num_output=85,
                        is_training=True,
                        reuse=False,
                        name="3D_module"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal: 
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    if reuse:
        print('Reuse is on!')
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(x, 1024, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(
            factor=.01, mode='FAN_AVG', uniform=True)
        pred_pose = slim.fully_connected(
            net,
            24 * 6,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='pose')
        pred_shape = slim.fully_connected(
            net,
            10,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='shape')
        pred_cam = slim.fully_connected(
            net,
            3,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='cam')
    variables = tf.contrib.framework.get_variables(scope)
    return pred_pose, pred_shape, pred_cam, variables


def Encoder_point(x,
                        num_output=2048,
                        is_training=True,
                        reuse=False,
                        name="3D_module"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal:
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    if reuse:
        print('Reuse is on!')
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(x, 1024, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(
            factor=.01, mode='FAN_AVG', uniform=True)
        net = slim.fully_connected(
            net,
            num_output,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='fc3')

    variables = tf.contrib.framework.get_variables(scope)
    return net, variables

def Encoder_fc3_dropout_translation(x,
                        num_output=3,
                        is_training=True,
                        reuse=False,
                        name="3D_module_translation"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal:
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    if reuse:
        print('Reuse is on!')
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(x, 1024, scope='fc1')
        #net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        #net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(
            factor=.01, mode='FAN_AVG', uniform=True)
        net = slim.fully_connected(
            net,
            num_output,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='fc3')

    variables = tf.contrib.framework.get_variables(scope)
    return net, variables

def get_encoder_fn_separate(model_type):
    """
    Retrieves diff encoder fn for image and 3D
    """
    encoder_fn = None
    threed_fn = None
    if 'resnet' in model_type:
        encoder_fn = Encoder_resnet
    else:
        print('Unknown encoder %s!' % model_type)
        exit(1)

    if 'fc3_dropout' in model_type:
        threed_fn = Encoder_fc3_dropout

    if encoder_fn is None or threed_fn is None:
        print('Dont know what encoder to use for %s' % model_type)
        # import ipdb
        # ipdb.set_trace()

    return encoder_fn, threed_fn


def Discriminator_separable_rotations(
        poses,
        shapes,
        weight_decay,
):
    """
    23 Discriminators on each joint + 1 for all joints + 1 for shape.
    To share the params on rotations, this treats the 23 rotation matrices
    as a "vertical image":
    Do 1x1 conv, then send off to 23 independent classifiers.

    Input:
    - poses: N x 23 x 1 x 9, NHWC ALWAYS!!
    - shapes: N x 10
    - weight_decay: float

    Outputs:
    - prediction: N x (1+23) or N x (1+23+1) if do_joint is on.
    - variables: tf variables
    """
    data_format = "NHWC"
    with tf.name_scope("Discriminator_sep_rotations", [poses, shapes]):
        with tf.variable_scope("D") as scope:
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.conv2d], data_format=data_format):
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv1')
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv2')
                    theta_out = []
                    for i in range(0, 23):
                        theta_out.append(
                            slim.fully_connected(
                                poses[:, i, :, :],
                                1,
                                activation_fn=None,
                                scope="pose_out_j%d" % i))
                    theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

                    # Do shape on it's own:
                    shapes = slim.stack(
                        shapes,
                        slim.fully_connected, [10, 5],
                        scope="shape_fc1")
                    shape_out = slim.fully_connected(
                        shapes, 1, activation_fn=None, scope="shape_final")
                    """ Compute joint correlation prior!"""
                    nz_feat = 1024
                    poses_all = slim.flatten(poses, scope='vectorize')
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc1")
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc2")
                    poses_all_out = slim.fully_connected(
                        poses_all,
                        1,
                        activation_fn=None,
                        scope="D_alljoints_out")
                    out = tf.concat([theta_out_all,
                                     poses_all_out, shape_out], 1)

            variables = tf.contrib.framework.get_variables(scope)
            return out, variables

def _batch_norm(data, is_training, bn_decay, scope):
    decay = bn_decay if bn_decay is not None else 0.5
    return tf.layers.batch_normalization(data, momentum=decay, training=is_training,
                                        beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                        gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                        name=scope)
def _variable_with_weight_decay(name, shape, wd, trainable=True):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
            name: name of the variable
            shape: list of ints
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

def _get_bias(shape, trainable=True):
    return tf.get_variable(name='biases', shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=trainable)

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    
    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:#, reuse=tf.AUTO_REUSE
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                        name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                            lambda: ema.apply([batch_mean, batch_var]),
                            lambda: tf.no_op())
        
        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        
        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed

def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.
    
    Args:
        inputs:      Tensor, 3D BLC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)

def _conv1d(inputs,num_output_channels,kernel_size,scope, stride=1,padding='SAME',weight_decay=0.0,bn=False,
            bn_decay=None,is_training=None, activation_fn=tf.nn.elu, trainable=True):
    """ 1D convolution with non-linear operation.
    Args:
        inputs: 3-D tensor variable BxLxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: int
        padding: 'SAME' or 'VALID'
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size, num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights', shape=kernel_shape, wd=weight_decay, trainable=trainable)
        outputs = tf.nn.conv1d(inputs, kernel,stride=stride,padding=padding)
        biases = _get_bias([num_output_channels], trainable)
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training, bn_decay=bn_decay, scope='bn')
        if activation_fn is None:
            print('No relu!')
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.
    
    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)
def _conv2d(inputs, num_output_channels,kernel_size,scope,stride=[1, 1],padding='SAME',weight_decay=0.0,bn=False, 
            bn_decay=None,is_training=None, activation_fn=tf.nn.elu, trainable=True):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',shape=kernel_shape, wd=weight_decay, trainable=trainable)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,[1, stride_h, stride_w, 1],padding=padding)
        biases = _get_bias([num_output_channels], trainable)
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
def conv2d(inputs, num_output_channels,kernel_size,scope,stride=[1, 1],padding='SAME',weight_decay=0.0,bn=False, 
            bn_decay=None,is_training=None, activation_fn=tf.nn.elu, trainable=True):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',shape=kernel_shape, wd=weight_decay, trainable=trainable)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,[1, stride_h, stride_w, 1],padding=padding)
        biases = _get_bias([num_output_channels], trainable)
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
def MLP_1d(features, mlp, is_training, bn_decay, bn, activation_fn=tf.nn.elu,scope='mlp1d', padding='VALID'):
    for i, num_out_channel in enumerate(mlp):
        features = _conv1d(features, num_out_channel, 1,
                                            padding=padding, stride=1,
                                            bn=bn, is_training=is_training,
                                            scope=scope+'_%d'%(i), bn_decay=bn_decay,activation_fn=activation_fn) 
    return features

def MLP_2d(features, mlp, is_training, bn_decay, bn, scope='mlp2d', padding='VALID'):
    for i, num_out_channel in enumerate(mlp):
        features = _conv2d(features, num_out_channel, [1,1],
                                            padding=padding, stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope= scope+'_%d'%(i), bn_decay=bn_decay) 
    return features

def coeff_generation(grouped_features, bn_decay, is_training=tf.cast(True,tf.bool), scope='attention', bn=True):
    with tf.variable_scope(scope) as sc:
        gac_par = [
            [32, 16], #MLP for xyz 
            [16, 16], #MLP for feature 
            [64] #hidden node of MLP for mergering 
        ]
        # grouped_features = tf.concat([grouped_xyz, grouped_features], axis=-1) #updata feature
        out_chal = grouped_features.get_shape()[-1].value
        coeff = MLP_1d(grouped_features, gac_par[2], is_training, bn_decay, bn, scope='conv')## map to a hiden feature space     
        # print(coeff)
        coeff = _conv1d(coeff, out_chal, 1, scope='conv1d', is_training=is_training, bn_decay=bn_decay, activation_fn=None)  #output coefficent
        # print(coeff)
        coeff = tf.nn.softmax(coeff, dim=2) #coefffient normalization
        # print(coeff)
        grouped_features = tf.multiply(coeff, grouped_features)
        # grouped_features = tf.reduce_sum(grouped_features, axis=[2], keep_dims=False)d
        return grouped_features
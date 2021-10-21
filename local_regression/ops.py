"""
TF util operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def keypoint3D_l1_loss_select(kp_gt, kp_pred, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 4))

        kp_pred = tf.reshape(kp_pred, (-1, 3))
        kp_pred_select = tf.gather(kp_pred[kp_gt[:, 3], :])

        res = tf.losses.absolute_difference(kp_gt[:, :3], kp_pred_select)

        return res

def keypoint3D_loss_select(kp_gt, kp_pred, batchsidx, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 4))
        kp_pred = tf.reshape(kp_pred, (-1, 3))
        skp_gt = tf.gather(kp_gt, batchsidx)
        skp_pred = tf.gather(kp_pred, batchsidx)
        svis = tf.cast(tf.reshape(skp_gt[:, 3],[-1]), tf.float32)
        td = skp_gt[:, 0:3]-skp_pred
        distsquare = tf.reduce_sum(tf.multiply(td,td), 1)
        res = tf.reduce_sum(tf.multiply(svis, distsquare))
        res = res / tf.reduce_sum(svis)
        return res

def keypoint3D_loss(kp_gt, kp_pred, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 4))
        kp_pred = tf.reshape(kp_pred, (-1, 3))
        vis = tf.cast(tf.reshape(kp_gt[:, 3],[-1]), tf.float32)
        td = kp_gt[:, 0:3]-kp_pred
        distsquare = tf.reduce_sum(tf.multiply(td,td), 1)
        res = tf.reduce_sum(tf.multiply(vis, distsquare))
        #res = res/tf.reduce_sum(vis)
        return res

def keypoint3D_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint3D_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 4))
        kp_pred = tf.reshape(kp_pred, (-1, 3))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 3], tf.float32), 1)
        res = tf.losses.absolute_difference(kp_gt[:, :3], kp_pred, weights=vis)
        return res

def keypoint_l1_loss_projection(kp_gt, kp_pred, name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    fx_d = 3.6667199999999998e+002
    cx_d = 2.5827199999999998e+002
    fy_d = 3.6667199999999998e+002
    cy_d = 2.0560100000000000e+002
    campro = tf.constant([ [fx_d,0,cx_d], [0,fy_d,cy_d], [0,0,1]])
    campro = tf.transpose(campro,[1,0])
    with tf.name_scope(name, "keypoint_l1_loss_projection", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 3))#x,y,z

        projkp = tf.matmul(kp_pred, campro)
        proi = tf.reshape(tf.div(projkp[:, 0], projkp[:, 2]),[-1,1])
        proj = tf.reshape(tf.div(projkp[:,1], projkp[:,2]),[-1,1])
        projkp2d = tf.concat([proi, proj],1)

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)

        res = tf.losses.absolute_difference(kp_gt[:, :2], projkp2d, weights=vis)
        return res

def keypoint_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)

        res = tf.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)
        return res

def compute_3d_loss_our(params_pred, params_gt):
    """
    Computes the l2 loss between 3D params pred and gt for those data that has_gt3d is True.
    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Inputs:
      params_pred: N x {226, 42}
      params_gt: N x {226, 42}
      # has_gt3d: (N,) bool
      has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope("3d_loss", [params_pred, params_gt]):
        # res = tf.losses.mean_squared_error(
        #     params_gt, params_pred) * 0.5
        res = tf.reduce_sum(tf.square(params_gt-params_pred))
        return res

def compute_3d_loss(params_pred, params_gt, has_gt3d):
    """
    Computes the l2 loss between 3D params pred and gt for those data that has_gt3d is True.
    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Inputs:
      params_pred: N x {226, 42}
      params_gt: N x {226, 42}
      # has_gt3d: (N,) bool
      has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope("3d_loss", [params_pred, params_gt, has_gt3d]):
        weights = tf.expand_dims(tf.cast(has_gt3d, tf.float32), 1)
        res = tf.losses.mean_squared_error(
            params_gt, params_pred, weights=weights) * 0.5
        return res


def align_by_pelvis(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    with tf.name_scope("align_by_pelvis", [joints]):
        left_id = 3
        right_id = 2
        pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        return joints - tf.expand_dims(pelvis, axis=1)

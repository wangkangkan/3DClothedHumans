import tensorflow as tf
import utils.tf_util as tf_util
import tensorflow.contrib.slim as slim
from transform_nets import input_transform_net, feature_transform_net
#from trainer_our_withoutR_pointdesc import nearestpoint_joint_fixedidx

def pointnet_feature(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope("pointnet") as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, 2048, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        # net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay)
        # net7 = net
        # net = tf_util.fully_connected(net, 2048, bn=True, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        #net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        #net = tf.nn.sigmoid(net, name="sigmoid")

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables

def pointfeature_patch(pointclass, classtensor, k, classnum,pointfeature, point_cloud):
    batch_size = pointclass.get_shape()[0].value
    num_point = pointclass.get_shape()[1].value
    Y1 = classtensor#(batch_size, classnum, 1) (16, 21, 1)
    Y2 = pointclass#(batch_size, num_points2, 1) (16, 336, 1)
    Y1T = tf.transpose(Y1, perm=[0, 2, 1])
    Y2T = tf.transpose(Y2, perm=[0, 2, 1])
    Y3 = tf.matmul(tf.multiply(Y1, Y1), tf.ones(tf.shape(Y2T))) + tf.matmul(tf.ones(tf.shape(Y1)),tf.multiply(Y2T, Y2T)) - tf.multiply(2.0, tf.matmul(Y1, Y2T))
    distance = tf.sqrt(Y3, name='match_relation_matrix')#(batch_size, num_points1, num_points2)
    neg_adj = -distance
    values, nn_idx = tf.nn.top_k(neg_adj,k=k)
    values = tf.abs(values)
    zeros = tf.zeros(tf.shape(values))
    ones = tf.ones(tf.shape(values))
    a = tf.equal(values,zeros)
    mask = tf.where(a,ones,zeros)
    # exit()
    mask = tf.expand_dims(mask,-1)
    print('mask',mask)
    idx_ = tf.range(batch_size) * num_point
    idx_ = tf.reshape(idx_, [batch_size, 1])
    nn_idx = tf.reshape(nn_idx,[batch_size,-1])
    print('value', values)
    print('nn_idx', nn_idx)
    IDX= nn_idx+idx_
    IDX = tf.reshape(IDX,[-1,1])
    print('IDX', IDX)
    # exit()
    pointfeature = tf.reshape(pointfeature,[batch_size*num_point,-1])
    feature_patch = tf.gather(pointfeature,IDX)
    feature_patch = tf.reshape(feature_patch,[batch_size,classnum,k,-1])
    # feature_patch = feature_patch*mask
    point_cloud = tf.reshape(point_cloud,[batch_size*num_point,-1])
    point_patch = tf.gather(point_cloud,IDX)
    point_patch = tf.reshape(point_patch,[batch_size,classnum,k,-1])
    # point_patch = point_patch*mask
    # patch_feat = tf.reduce_max(feature_patch,axis=2)
    #print(patch_feat)
    return feature_patch, point_patch

def eachpointfeature(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    # print(point_cloud)
    with tf.variable_scope("eachpointfeat") as scope:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        point_feat = tf.expand_dims(net_transformed, [2])
        # print(point_feat)

        net = tf_util.conv2d(point_feat, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        pointfeature = net
        global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                         padding='VALID', scope='maxpool')
        global_feat = tf.reshape(global_feat, [batch_size, -1])
    variables = tf.contrib.framework.get_variables(scope)

    return pointfeature, global_feat, variables

def get_model(point_cloud, classnum, classtensor, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    print('ppoint_cloud', point_cloud)
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    # print(point_cloud)
    with tf.variable_scope("pointnet") as scope:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1,3],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        point_feat = tf.expand_dims(net_transformed, [2])
        print('point_feat', point_feat)

        net = tf_util.conv2d(point_feat, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 512, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)

        pointfeature = net
        print('net', pointfeature)
        global_feat = tf_util.max_pool2d(net, [num_point,1],
                                        padding='VALID', scope='maxpool')
        print('global_feat',global_feat)
        # exit()

        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        # print(global_feat_expand)
        concat_feat = tf.concat([point_feat, global_feat_expand],3)
        # print(concat_feat)

        net = tf_util.conv2d(concat_feat, 512, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv6', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv7', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv8', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv9', bn_decay=bn_decay)

        net = tf_util.conv2d(net, classnum, [1,1],
                            padding='VALID', stride=[1,1], activation_fn=None,
                            scope='conv10')
        pointclass = tf.squeeze(net, [2]) # BxNxC

    # with tf.variable_scope("pointfeat") as scopefeat:
    #     pointfeature = tf_util.conv2d(pointfeature, 1024, [1, 1],
    #                                   padding='VALID', stride=[1, 1],
    #                                   bn=True, is_training=is_training,
    #                                   scope='convfeat1', bn_decay=bn_decay)
    #     pointfeature = tf_util.conv2d(pointfeature, 1024, [1, 1],
    #                                   padding='VALID', stride=[1, 1],
    #                                   bn=True, is_training=is_training,
    #                                   scope='convfeat2', bn_decay=bn_decay)
    #variablesfeat = tf.contrib.framework.get_variables(scopefeat)

    # pointfeature, global_feat, variablesfeat = eachpointfeature(point_cloud, is_training=is_training)

    variables = tf.contrib.framework.get_variables(scope)

    # variables.extend(variablesfeat)

    pointclassSM = tf.nn.softmax(pointclass)
    pointclassAM = tf.argmax(pointclassSM, 2)
    pointclassAM = tf.expand_dims(pointclassAM,-1)
    pointclassAM = tf.cast(pointclassAM,tf.float32)
    patch_feat, patch_point = pointfeature_patch(pointclassAM, classtensor, 16, classnum, pointfeature, point_cloud)
    # print(patch_feat)
    # exit()

    # patch_feat = tf.reshape(patch_feat,[batch_size,-1])
    # patch_point = tf.reshape(patch_point,[batch_size,-1])

    # print(pointclassAM)
    # print(pointfeature)
    # print(pointclass)
    # global_feat = tf.reshape(global_feat, [batch_size, -1])
    #patch_feat = tf.concat([patch_feat, global_feat], 1)
    return patch_feat, patch_point, pointclass, variables


def get_model_withoutclass(point_cloud, pointfeature, classnum, classtensor, pointclass, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    # print('ppoint_cloud', point_cloud)
    # batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    # end_points = {}
    # # print(point_cloud)
    # with tf.variable_scope("pointnet") as scope:
    #     with tf.variable_scope('transform_net1') as sc:
    #         transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    #     point_cloud_transformed = tf.matmul(point_cloud, transform)
    #     input_image = tf.expand_dims(point_cloud_transformed, -1)

    #     net = tf_util.conv2d(input_image, 64, [1,3],
    #                         padding='VALID', stride=[1,1],
    #                         bn=True, is_training=is_training,
    #                         scope='conv1', bn_decay=bn_decay)
    #     net = tf_util.conv2d(net, 64, [1,1],
    #                         padding='VALID', stride=[1,1],
    #                         bn=True, is_training=is_training,
    #                         scope='conv2', bn_decay=bn_decay)

    #     with tf.variable_scope('transform_net2') as sc:
    #         transform = feature_transform_net(net, is_training, bn_decay, K=64)
    #     end_points['transform'] = transform
    #     net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    #     point_feat = tf.expand_dims(net_transformed, [2])
    #     print('point_feat', point_feat)

    #     net = tf_util.conv2d(point_feat, 64, [1,1],
    #                         padding='VALID', stride=[1,1],
    #                         bn=True, is_training=is_training,
    #                         scope='conv3', bn_decay=bn_decay)
    #     net = tf_util.conv2d(net, 128, [1,1],
    #                         padding='VALID', stride=[1,1],
    #                         bn=True, is_training=is_training,
    #                         scope='conv4', bn_decay=bn_decay)
    #     net = tf_util.conv2d(net, 512, [1,1],
    #                         padding='VALID', stride=[1,1],
    #                         bn=True, is_training=is_training,
    #                         scope='conv5', bn_decay=bn_decay)

    #     pointfeature = net
    #     # print('net', pointfeature)
    #     # global_feat = tf_util.max_pool2d(net, [num_point,1],
    #     #                                 padding='VALID', scope='maxpool')
    #     # print('global_feat',global_feat)
    #     # # exit()

    #     # global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    #     # # print(global_feat_expand)
    #     # concat_feat = tf.concat([point_feat, global_feat_expand],3)
    #     # # print(concat_feat)

    #     # net = tf_util.conv2d(concat_feat, 512, [1,1],
    #     #                     padding='VALID', stride=[1,1],
    #     #                     bn=True, is_training=is_training,
    #     #                     scope='conv6', bn_decay=bn_decay)
    #     # net = tf_util.conv2d(net, 256, [1,1],
    #     #                     padding='VALID', stride=[1,1],
    #     #                     bn=True, is_training=is_training,
    #     #                     scope='conv7', bn_decay=bn_decay)
    #     # net = tf_util.conv2d(net, 128, [1,1],
    #     #                     padding='VALID', stride=[1,1],
    #     #                     bn=True, is_training=is_training,
    #     #                     scope='conv8', bn_decay=bn_decay)
    #     # net = tf_util.conv2d(net, 128, [1,1],
    #     #                     padding='VALID', stride=[1,1],
    #     #                     bn=True, is_training=is_training,
    #     #                     scope='conv9', bn_decay=bn_decay)

    #     # net = tf_util.conv2d(net, classnum, [1,1],
    #     #                     padding='VALID', stride=[1,1], activation_fn=None,
    #     #                     scope='conv10')
    #     # pointclass = tf.squeeze(net, [2]) # BxNxC

    # with tf.variable_scope("pointfeat") as scopefeat:
    #     pointfeature = tf_util.conv2d(pointfeature, 1024, [1, 1],
    #                                   padding='VALID', stride=[1, 1],
    #                                   bn=True, is_training=is_training,
    #                                   scope='convfeat1', bn_decay=bn_decay)
    #     pointfeature = tf_util.conv2d(pointfeature, 1024, [1, 1],
    #                                   padding='VALID', stride=[1, 1],
    #                                   bn=True, is_training=is_training,
    #                                   scope='convfeat2', bn_decay=bn_decay)
    #variablesfeat = tf.contrib.framework.get_variables(scopefeat)

    # pointfeature, global_feat, variablesfeat = eachpointfeature(point_cloud, is_training=is_training)

    variables = None
    # variables = tf.contrib.framework.get_variables(scope)

    # variables.extend(variablesfeat)

    # pointclassSM = tf.nn.softmax(pointclass)
    # pointclassAM = tf.argmax(pointclassSM, 2)
    # pointclassAM = tf.expand_dims(pointclass,-1)
    pointclassAM = tf.cast(pointclass,tf.float32)
    patch_feat, patch_point = pointfeature_patch(pointclassAM, classtensor, 16, classnum, pointfeature, point_cloud)
    # print(patch_feat)
    # exit()

    # patch_feat = tf.reshape(patch_feat,[batch_size,-1])
    # patch_point = tf.reshape(patch_point,[batch_size,-1])

    # print(pointclassAM)
    # print(pointfeature)
    # print(pointclass)
    # global_feat = tf.reshape(global_feat, [batch_size, -1])
    #patch_feat = tf.concat([patch_feat, global_feat], 1)
    return patch_feat, patch_point, pointclass, variables



def get_model_class(point_cloud, classnum, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    # print(point_cloud)
    with tf.variable_scope("pointnet") as scope:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1,3],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        point_feat = tf.expand_dims(net_transformed, [2])
        print(point_feat)

        net = tf_util.conv2d(point_feat, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)
        global_feat = tf_util.max_pool2d(net, [num_point,1],
                                        padding='VALID', scope='maxpool')
        print(global_feat)

        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        print(global_feat_expand)
        concat_feat = tf.concat([point_feat, global_feat_expand],3)
        print(concat_feat)

        net = tf_util.conv2d(concat_feat, 512, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv6', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv7', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv8', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv9', bn_decay=bn_decay)

        net = tf_util.conv2d(net, classnum, [1,1],
                            padding='VALID', stride=[1,1], activation_fn=None,
                            scope='conv10')
        pointclass = tf.squeeze(net, [2]) # BxNxC
    variables = tf.contrib.framework.get_variables(scope)
    return pointclass, variables

def pointnet_feature_classification(point_cloud, classnum, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope("pointnet") as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        pointfeats = net
        pointfeats = tf.squeeze(pointfeats, 2)#B*N*1024

        net = tf_util.conv2d(net, 512, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        net = tf_util.conv2d(net, 256, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)

        pointclass = tf_util.conv2d(net, classnum, [1, 1],
                             padding='VALID', stride=[1, 1],activation_fn=None,
                             bn=True, is_training=is_training,
                             scope='conv7', bn_decay=bn_decay)
        pointclass = tf.squeeze(pointclass, 2)#B*N*classnum

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return pointfeats, pointclass, variables

def pointnet_feature_latent(img_feat, point_cloud, latentsize, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope("pointnet") as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf.concat([img_feat, net], 1)
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net7 = net
        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, latentsize, activation_fn=None, scope='fc3')

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables

def pointnet_feature2(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope("pointnet") as scope:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        net = tf_util.conv2d(net_transformed, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        # net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay)
        # net7 = net
        # net = tf_util.fully_connected(net, 2048, bn=True, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        #net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        #net = tf.nn.sigmoid(net, name="sigmoid")

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables

def get_model0(point_cloud, is_training, reuse, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope("pointnet", reuse=reuse) as scope:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        net = tf_util.conv2d(net_transformed, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp2')
        net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')
        variables = tf.contrib.framework.get_variables(scope)

    return net, variables

def pointnet_discriminator(point_cloud, is_training, reuse, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope("pointnet", reuse=reuse) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net7 = net
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')
        #net = tf.nn.sigmoid(net, name="sigmoid")

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables

def discriminator_pointfeature(point_cloud, is_training, name, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope(name) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])

        return net

def discriminatorwithlocalfeature(new_points, mlp, is_training, reuse, bn=True, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = new_points.get_shape()[0].value
    num_point = new_points.get_shape()[1].value

    with tf.variable_scope("discriminatorpointnet", reuse=reuse) as scope:
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

        new_points = tf.reshape(new_points, [batch_size, num_point, 1,-1])
        #new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        #new_points = tf.squeeze(new_points, [2])

        # with slim.arg_scope(
        #         [slim.conv2d],
        #         weights_regularizer=slim.l2_regularizer(0.0001)):
        #     with slim.arg_scope([slim.conv2d], data_format="NHWC"):
        #         net = slim.conv2d(new_points, 512, [1, 1], scope='D_conv1')
        #         net = slim.conv2d(net, 512, [1, 1], scope='D_conv2')
        #
        theta_out = []
        for i in range(0, 24):
            # poseout = slim.fully_connected(
            #         new_points[:, i, :, :],
            #         128,
            #         activation_fn=None,
            #         scope="pose_out_0%d" % i)
            # poseout = slim.fully_connected(
            #     poseout,
            #     1,
            #     activation_fn=None,
            #     scope="pose_out_1%d" % i)
            # theta_out.append(poseout)
            theta_out.append(
                slim.fully_connected(
                    new_points[:, i, :, :],
                    1,
                    activation_fn=None,
                    scope="pose_out_j%d" % i))
        theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))
        if batch_size == 1:
            theta_out_all = tf.expand_dims(theta_out_all, 0)

        net = slim.flatten(new_points, scope='vectorize')
        # temp = mlp[-1] * num_point
        # net = tf.reshape(new_points, [-1, temp])
        nz_feat = 1024
        net = slim.fully_connected(
            net, nz_feat, scope="D_alljoints_fc1")
        net = slim.fully_connected(
            net, nz_feat, scope="D_alljoints_fc2")
        poses_all_out = slim.fully_connected(
            net,
            1,
            activation_fn=None,
            scope="D_alljoints_out")

        # new_points = tf.squeeze(new_points, [2])
        # new_points = tf.reduce_max(new_points, axis=[1], keep_dims=True, name='maxpool')
        # net = tf.squeeze(new_points, [1])
        # net = slim.fully_connected(
        #     net, nz_feat, scope="D_global_fc1")
        # net = slim.fully_connected(
        #     net, nz_feat, scope="D_global_fc2")
        # global_out = slim.fully_connected(
        #     net,
        #     1,
        #     activation_fn=tf.nn.sigmoid,
        #     scope="D_global_out")

        #
        # net = slim.flatten(net, scope='vectorize')
        #
        # net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay)
        #
        # net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        # poses_all_out = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')
        #
        net = tf.concat([theta_out_all,
                         poses_all_out], 1)

        # with slim.arg_scope(
        #         [slim.conv2d],
        #         weights_regularizer=slim.l2_regularizer(0.0001)):
        #     with slim.arg_scope([slim.conv2d], data_format="NHWC"):
        #         net = slim.conv2d(new_points, 512, [1, 1], scope='D_conv1')
        #         net = slim.conv2d(net, 512, [1, 1], scope='D_conv2')
        #
        # net = slim.flatten(new_points, scope='vectorize')
        # # temp = mlp[-1] * num_point
        # # net = tf.reshape(new_points, [-1, temp])
        # #
        # net = tf_util.fully_connected(net, 512, bn=bn, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay)
        #
        # net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        # net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc3')
        #
        # net = tf.nn.sigmoid(net, name="sigmoid")

        # with slim.arg_scope(
        #         [slim.conv2d, slim.fully_connected],
        #         weights_regularizer=slim.l2_regularizer(0.0001)):
        #         with slim.arg_scope([slim.conv2d], data_format="NHWC"):
        #             net = slim.conv2d(new_points, 32, [1, 1], scope='D_conv1')
        #             net = slim.conv2d(net, 32, [1, 1], scope='D_conv2')
        #             net = slim.flatten(net, scope='vectorize')
        #             nz_feat = 1024
        #             net = slim.fully_connected(
        #                 net, nz_feat, scope="D_alljoints_fc1")
        #             net = slim.fully_connected(
        #                 net, nz_feat, scope="D_alljoints_fc2")
        #             net = slim.fully_connected(
        #                 net,
        #                 1,
        #                 activation_fn=None,
        #                 scope="D_alljoints_out")

        variables = tf.contrib.framework.get_variables(scope)

        return net, variables

def pointnet2feature(points, joints, k, thres, mlp, is_training, reuse, bn=True, bn_decay=None):

    num_points1 = joints.get_shape()[1].value
    num_points2 = points.get_shape()[1].value

    Y1 = joints  # (batch_size, num_points1, 3)
    Y2 = points  # (batch_size, num_points2, 3)
    Y1T = tf.transpose(Y1, perm=[0, 2, 1])
    Y2T = tf.transpose(Y2, perm=[0, 2, 1])
    Y3 = tf.matmul(tf.multiply(Y1, Y1), tf.ones(tf.shape(Y2T))) + tf.matmul(tf.ones(tf.shape(Y1)),
                                                                            tf.multiply(Y2T, Y2T)) - tf.multiply(
        2.0, tf.matmul(Y1, Y2T))
    distance = tf.sqrt(Y3, name='match_relation_matrix')  # (batch_size, num_points1, num_points2)

    neg_adj = -distance
    dis_adj, nn_idx = tf.nn.top_k(neg_adj, k=k)
    dislabel = tf.cast(tf.greater(dis_adj, -thres), dtype=tf.int32)#less than thres
    newidx = tf.multiply(nn_idx,dislabel)#idx of which distance larger than thres is set to 0

    firstidx = tf.reshape(newidx[:,:,0],[-1,num_points1,1])
    firstidx = tf.tile(firstidx, [1, 1, k])
    firstidx = tf.multiply(firstidx, 1-dislabel)#operate where idx is 0
    newidx = newidx+firstidx#idx 0 is replaced with the first idx

    point_cloud_shape = points.get_shape()
    num_points = point_cloud_shape[1].value
    batch_size = point_cloud_shape[0].value
    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])
    point_cloud_flat = tf.reshape(points, [-1, 3])
    new_points = tf.gather(point_cloud_flat, newidx + idx_)
    newp = new_points
    new_points -= tf.tile(tf.expand_dims(joints, 2), [1, 1, k, 1])

    with tf.variable_scope("pointnet2", reuse=reuse) as scope:
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

    new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')

    new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])

    findpointnum = tf.reduce_sum(dislabel, axis=2)
    ptnumlabel = tf.cast(tf.greater(findpointnum, 0), dtype=tf.int32)
    labelexpand = tf.cast(tf.tile(tf.expand_dims(ptnumlabel, -1), [1, 1, mlp[-1]]), dtype=tf.float32)
    net = tf.multiply(new_points, labelexpand)

    variables = tf.contrib.framework.get_variables(scope)

    return net, variables

def partpointfeature(pointfeature, joint_nnidx, batch_size, num_stage):

    batchsample = batch_size*num_stage*2
    nn_idx = tf.tile(joint_nnidx,[batchsample,1,1])

    point_cloud_shape = pointfeature.get_shape()
    num_points = point_cloud_shape[1].value
    featdim = point_cloud_shape[2].value
    idx_ = tf.range(batchsample) * num_points
    idx_ = tf.reshape(idx_, [batchsample, 1, 1])
    point_cloud_flat = tf.reshape(pointfeature, [-1, featdim])
    points_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    return points_neighbors

def discriminatorwithlocalfeatureandglobalfeature(new_points, joint_nnidx, batchsize, num_stage, mlp, is_training, reuse, bn=True, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = new_points.get_shape()[0].value
    new_points = tf.expand_dims(new_points, 1)  # Bx1xNx3
    num_point = new_points.get_shape()[2].value

    with tf.variable_scope("pointnet", reuse=reuse) as scope:
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay)

        #new_points = tf.reshape(new_points, [batch_size, -1])#B*1*N*C
        new_points = tf.reshape(new_points, [batch_size, num_point, -1])#B*N*C
        #new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        #new_points = tf.squeeze(new_points, [2])

        # with slim.arg_scope(
        #         [slim.conv2d],
        #         weights_regularizer=slim.l2_regularizer(0.0001)):
        #     with slim.arg_scope([slim.conv2d], data_format="NHWC"):
        #         net = slim.conv2d(new_points, 512, [1, 1], scope='D_conv1')
        #         net = slim.conv2d(net, 512, [1, 1], scope='D_conv2')
        #
        neighborpointfeature = partpointfeature(new_points, joint_nnidx, batchsize, num_stage)#B*24*idx*C
        neighborpointfeature = tf.reshape(neighborpointfeature, [batch_size, 24, 1, -1])
        theta_out = []
        for i in range(0, 24):
            # poseout = slim.fully_connected(
            #         new_points[:, i, :, :],
            #         128,
            #         activation_fn=None,
            #         scope="pose_out_0%d" % i)
            # poseout = slim.fully_connected(
            #     poseout,
            #     1,
            #     activation_fn=None,
            #     scope="pose_out_1%d" % i)
            # theta_out.append(poseout)
            theta_out.append(
                slim.fully_connected(
                    neighborpointfeature[:, i, :, :],
                    1,
                    activation_fn=None,
                    scope="pose_out_j%d" % i))
        theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))
        if batch_size == 1:
            theta_out_all = tf.expand_dims(theta_out_all, 0)

        net = slim.flatten(new_points, scope='vectorize')
        # temp = mlp[-1] * num_point
        # net = tf.reshape(new_points, [-1, temp])
        nz_feat = 1024
        net = slim.fully_connected(
            net, nz_feat, scope="D_alljoints_fc1")
        net = slim.fully_connected(
            net, nz_feat, scope="D_alljoints_fc2")
        poses_all_out = slim.fully_connected(
            net,
            1,
            activation_fn=None,
            scope="D_alljoints_out")

        net = tf.concat([theta_out_all,
                         poses_all_out], 1)

        variables = tf.contrib.framework.get_variables(scope)

        return net, variables

def discriminatorwithlocalfeature_onegroup(new_points, mlp, is_training, reuse, bn=True, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = new_points.get_shape()[0].value
    new_points = tf.expand_dims(new_points, 1)#Bx1xNx3
    num_point = new_points.get_shape()[2].value

    with tf.variable_scope("pointnet", reuse=reuse) as scope:
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

        new_points = tf.reshape(new_points, [batch_size, -1])
        #new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        #new_points = tf.squeeze(new_points, [2])

        # with slim.arg_scope(
        #         [slim.conv2d],
        #         weights_regularizer=slim.l2_regularizer(0.0001)):
        #     with slim.arg_scope([slim.conv2d], data_format="NHWC"):
        #         net = slim.conv2d(new_points, 512, [1, 1], scope='D_conv1')
        #         net = slim.conv2d(net, 512, [1, 1], scope='D_conv2')
        #

        # theta_out = slim.fully_connected(
        #         new_points[:, 0, :, :],
        #         1,
        #         activation_fn=None,
        #         scope="pose_out_j0")
        # theta_out_all = tf.reshape(theta_out, [-1, 1])

        # temp = mlp[-1] * num_point
        # net = tf.reshape(new_points, [-1, temp])
        nz_feat = 1024
        net = slim.fully_connected(
            new_points, nz_feat, scope="D_alljoints_fc1")
        net = slim.fully_connected(
            net, nz_feat, scope="D_alljoints_fc2")
        poses_all_out = slim.fully_connected(
            net,
            1,
            activation_fn=None,
            scope="D_alljoints_out")

        # net = tf.concat([theta_out_all,
        #                  poses_all_out], 1)
        net = poses_all_out
        variables = tf.contrib.framework.get_variables(scope)

        return net, variables

def extractlocalfeature(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope("pointnet") as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        net = tf.squeeze(net, [2])

        # weights = tf_util._variable_with_weight_decay('weights',
        #                                       shape=[num_point, 1024],
        #                                       use_xavier=True,
        #                                       stddev=1e-3,
        #                                       wd=0.0)
        # weights = tf.reshape(weights, [1, num_point, 1024])
        # batchweights = tf.tile(weights, [batch_size, 1, 1])
        #
        # net = tf.multiply(net, batchweights)
        # net = tf.reduce_sum(net, 1)

        max_net = tf.reduce_max(net, axis=[1], keep_dims=True, name='maxpool')
        avg_net = tf.reduce_mean(net, axis=[1], keep_dims=True, name='avgpool')
        net = tf.concat([avg_net, max_net], axis=-1)

        # Symmetric function: max pooling
        # net = tf_util.max_pool2d(net, [num_point,1],
        #                          padding='VALID', scope='maxpool')
        # net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        # net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay)
        # net7 = net
        # net = tf_util.fully_connected(net, 2048, bn=True, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        #net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        #net = tf.nn.sigmoid(net, name="sigmoid")

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables

def extractfeature_onrawpoints(point_cloud, is_training, reuse, name, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        net = tf.squeeze(net)
        if batch_size == 1:
            net = tf.expand_dims(net, 0)

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables

def extractglobalfeature_onmatchcorrelation(point_cloud, correlation_map, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_feature = tf.concat([point_cloud, correlation_map], 2)
    input_feature = tf.reshape(input_feature, [batch_size, num_point,1,-1])# input_feature BxNxM, input_image BxNx1xM

    with tf.variable_scope("extractglobalfeatureonmatchcorrelation") as scope:
        # Point functions (MLP implemented as conv2d)

        net = tf_util.conv2d(input_feature, 64, [1,1],
                              padding='VALID', stride=[1,1],
                              bn=True, is_training=is_training,
                              scope='conv1', bn_decay=bn_decay)

        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)

        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        # net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
        #                               scope='fc1', bn_decay=bn_decay)
        # net7 = net
        # net = tf_util.fully_connected(net, 2048, bn=True, is_training=is_training,
        #                               scope='fc2', bn_decay=bn_decay)
        # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
        #                       scope='dp1')
        #net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
        #net = tf.nn.sigmoid(net, name="sigmoid")

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
        return net, variables

def feature_extract(point_cloud, is_training, reuse, outdim, name, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, outdim, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
    return net, variables

def match_embedding(match_relation, is_training, reuse, outdim, name, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(match_relation, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, outdim, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
    return net, variables

def ournetwork(point_cloud, is_training, reuse, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope("pointnet", reuse=reuse) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net7 = net
        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 85, activation_fn=None, scope='fc3')
        # net = tf.nn.sigmoid(net, name="sigmoid")

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
    return net, variables

def ournetwork_outdim(point_cloud, is_training, reuse, outdim, name, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv2', bn_decay=bn_decay)
        # net2 = net
        # net = tf_util.conv2d(net, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, outdim, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
    return net, variables

def ournetwork_6d(point_cloud, is_training, reuse, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    input_image = tf.expand_dims(point_cloud, -1)  # input_image BxNx3x1

    with tf.variable_scope("pointnet", reuse=reuse) as scope:
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net1 = net
        net = tf_util.conv2d(net, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        # net2 = net
        net = tf_util.conv2d(net, 64, [1,2],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net3 = net
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net4 = net
        net = tf_util.conv2d(net, 256, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        net5 = net

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')
        net6 = net
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net7 = net
        net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 85, activation_fn=None, scope='fc3')
        # net = tf.nn.sigmoid(net, name="sigmoid")

        variables = tf.contrib.framework.get_variables(scope)
        # # print(variables)
        # for val in variables:
        #     print(val)
        # exit()
    return net, variables


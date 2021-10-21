from . import rescale

import tensorflow as tf
import scipy.sparse
import numpy as np
import os

class DecodeModel(object):
    """
    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.
        nz: Size of latent variable.
        F_0: Number of graph input features.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    U: Upsampling matrix
       
    """
    #which_loss l1 or l2 defaut=l1
    def __init__(self, L, D, U, F, K, p, nz, which_loss='l1', F_0=1, filter='chebyshev5', brelu='b1relu',
                pool='poolwT',unpool='poolwT', regularization=0, dropout=0, batch_size=100,
                dir_name=' '):
        # super(DecodeModel, self).__init__()
        
        # Keep the useful Laplacians only. May be zero.
        self.M_0 = L[0].shape[0]
        # Store attributes and bind operations.
        self.L, self.D, self.U, self.F, self.K, self.p, self.nz, self.F_0 = L, D, U, F, K, p, nz, F_0
        self.which_loss = which_loss
        self.regularization, self.dropout = regularization, dropout
        self.batch_size = batch_size
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.unpool = getattr(self, unpool)

        self.regularizers = []

        self.dir_name = dir_name

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = rescale.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x) # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2,0,1]) # N x Mp x Fin

        return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _decode_full_4layers(self, x, reuse=tf.AUTO_REUSE, use_res_block=False, is_training=True):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            N = x.get_shape()[0]
            M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            x = tf.reshape(x, [N, -1])
            print('decode:\n',x.shape)
            with tf.variable_scope('fc2'):
                x = self.fc(x, int(self.p[-1]*self.F[-1]))            # N x MF
                print('fc:',x.shape)

            x = tf.reshape(x, [int(N), int(self.p[-1]), int(self.F[-1])])  # N x M x F
            print('reshape:',x.shape)

            for i in range(0, len(self.F)):
                with tf.variable_scope('upconv{}'.format(i+1)):

                    if not use_res_block:
                        with tf.name_scope('filter'):
                            x = self.filter(x, self.L[0], self.F[-i-1], self.K[0])
                            print('filter:',x.shape)
                        with tf.name_scope('bias_relu'):
                            x = self.brelu(x)
                            print('brelu:',x.shape)
                    else:
                        x = self.gcn_res_block_4layers(x, i, name='gcn_res_block_{:d}'.format(i), reuse=reuse, is_training=is_training)

            with tf.name_scope('outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])
                # x = tf.tanh(x)
                print('refilter:',x.shape)
        return x

    def fc_bn_moudle(self, x, channel=128, is_training=True, bn=True):
        x = tf.layers.dense(x, channel)
        x = tf.nn.relu(x)
        if bn:
            x = tf.layers.batch_normalization(x, training=is_training)

        return x

    def fc_skip(self, x, is_training=True):
        with tf.variable_scope('fc_skip', reuse=tf.AUTO_REUSE):
            # pass
            x = tf.layers.batch_normalization(x, training=is_training)
            x0 = x
            x = self.fc_bn_moudle(x, is_training=is_training)
            x = self.fc_bn_moudle(x, is_training=is_training)
            x1 = x
            x = tf.concat([x0, x], axis=-1)
            x = self.fc_bn_moudle(x, is_training=is_training)
            x = self.fc_bn_moudle(x, is_training=is_training)
            x = tf.concat([x0, x1, x], axis=-1)

            x = tf.layers.dense(x, 3)
            x = tf.reshape(x, [-1, 6890, 3])
            return x

    def group_normalizaton(self, x, is_training, name, norm_type='group', G=8, eps=1e-5, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            if norm_type == None:
                output = x 
            elif norm_type == 'batch':
                output = tf.contrib.layers.atch_norm(
                    x, center=True, scale=True, decay=0.999,
                    is_training=is_training, updates_collections=None
                )
            elif norm_type == 'group':
                # tranpose: [bs, v, c] to [bs, c, v] following the GraphCMR paper
                x = tf.transpose(x, [0, 2, 1])
                N, C, V = x.get_shape().as_list() # v is num of verts
                G = min(G, C)
                x = tf.reshape(x, [-1, G, C // G, V])
                mean, var = tf.nn.moments(x, axes=[2, 3], keep_dims=True)
                x = (x -mean) / tf.sqrt(var + eps)
                # per channel gamma and beta
                gamma = tf.get_variable('gamma', shape=[C], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
                beta  =tf.get_variable('beta', shape=[C], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                gamma = tf.reshape(gamma, [1, C, 1])
                beta = tf.reshape(beta, [1, C, 1])

                output = tf.reshape(x, [-1, C, V]) * gamma + beta

                output = tf.transpose(output, [0, 2, 1])

            else:
                raise NotImplementedError

        return output

    def gcn_res_block(self, x_in, i, name, reuse=True, is_training=True):
        with tf.variable_scope(name, reuse=reuse):
            x = self.group_normalizaton(x_in, is_training=is_training, name='group_norm', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_linear_1'):
                x = self.filter(x, self.L[0], self.F[-i-1] // 2, 1)
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_1', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_conv'):
                x = self.filter(x, self.L[0], self.F[-i-1] // 2, self.K[0])
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_2', reuse=reuse)

            x = tf.nn.relu(x)
            
            with tf.variable_scope('graph_linear_2'):
                x = self.filter(x, self.L[0], self.F[-i-1], 1)
            
            channel_in = x_in.get_shape()[-1]
            channel_out = x.get_shape()[-1]
            if channel_in != channel_out:
                with tf.variable_scope('graph_linear_input'):
                    x_in = self.filter(x_in, self.L[0], channel_out, 1)

            # skip connection
            x = x + x_in

        return x
    
    def gcn_res_block_4layers(self, x_in, i, name, reuse=False, is_training=False):
        with tf.variable_scope(name, reuse=reuse):

            with tf.name_scope('unpooling'):
                x_in = self.unpool(x_in, self.U[-i-1])
                print('unpool:',x_in.shape)

            x = self.group_normalizaton(x_in, is_training=is_training, name='group_norm', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_linear_1'):
                x = self.filter(x, self.L[-i-2], self.F[-i-1] // 2, 1)
                print('filter:',x.shape)

            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_1', reuse=reuse)

            x = tf.nn.relu(x)

            with tf.variable_scope('graph_conv'):
                x = self.filter(x, self.L[-i-2], self.F[-i-1] // 2, self.K[0])
            x = self.group_normalizaton(x, is_training=is_training, name='group_norm_2', reuse=reuse)

            x = tf.nn.relu(x)
            
            with tf.variable_scope('graph_linear_2'):
                x = self.filter(x, self.L[-i-2], self.F[-i-1], 1)
            
            channel_in = x_in.get_shape()[-1]
            channel_out = x.get_shape()[-1]
            if channel_in != channel_out:
                with tf.variable_scope('graph_linear_input'):
                    x_in = self.filter(x_in, self.L[-i-2], channel_out, 1)

            # skip connection
            x = x + x_in

        return x

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

def store_sp_matrix(list_of_matrix,  name):
    """
    Param:
        list_of_matrix: A list of sparse matrix.
        name: The name of matrix needed to store.
    
    """
    dir_name = os.getcwd()
    dir_name = os.path.join(dir_name, 'matrix', name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for i in range(len(list_of_matrix)):
        assert(scipy.sparse.issparse(list_of_matrix[i]))

        abs_dir_name = dir_name + '/' + str(i) + '.npz'
        scipy.sparse.save_npz(abs_dir_name, list_of_matrix[i])




def read_sp_matrix(name):
    """
    Param:
        name: The name of matrix needed to read.

    Return:
        A list of sparse matrix.

    """
    dir_name = os.path.dirname(os.path.realpath(__file__)) 
    dir_name = os.path.join(dir_name, name)
    sp_matrix = []
    list_dir = os.listdir(dir_name)
    for i in range(len(list_dir)):
        dir_name_npz = dir_name + '/' + str(i) + '.npz'
        sparse_matrix_variable = scipy.sparse.load_npz(dir_name_npz)
        sp_matrix.append(sparse_matrix_variable)

    return sp_matrix
import numpy as np
#import cPickle as pickle
import pickle as pickle

import tensorflow as tf
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(object):
    def __init__(self, pkl_path, joint_type='cocoplus', dtype=tf.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'rb') as f:

            dd = pickle.load(f, encoding='iso-8859-1')
            # else:
            #     assert(version[0] == '2')
            #     dd = pickle.load(f)
        self.v_face = dd['f']
        self.verts = dd['v_template']
        # Mean template vertices
        self.v_template = tf.Variable(
            undo_chumpy(dd['v_template']),
            name='v_template',
            dtype=dtype,
            trainable=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(
            shapedir, name='shapedirs', dtype=dtype, trainable=False)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = tf.Variable(
            dd['J_regressor'].T.todense(),
            name="J_regressor",
            dtype=dtype,
            trainable=False)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(
            posedirs, name='posedirs', dtype=dtype, trainable=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = tf.Variable(
            undo_chumpy(dd['weights']),
            name='lbs_weights',
            dtype=dtype,
            trainable=False)

        ########## 16535 point #################
        edge = np.loadtxt('edge_16535.txt', dtype=np.int)
        self.edge = tf.constant(edge, dtype=tf.int32, shape=[9645, 2])

        #################### calculate rotation matrix ######################
        cos_value = np.cos(np.pi)
        sin_value = np.sin(np.pi)

        self.rotation = tf.constant(
            [
                [1., 0., 0.],
                [0., cos_value, -1*sin_value],
                [0., 1*sin_value, cos_value]
            ],
            dtype=tf.float32,
            shape=[1, 3, 3],
            name="rotation_matrix"
        )

    def mesh_x(self, verts):
        # pass
        ##### 6890 -> 1723
        # verts = tf.gather(verts, indices=self.idx_1723_tf, axis=1)


        verts_0 = tf.gather(verts, indices=self.edge[:, 0], axis=1)
        verts_1 = tf.gather(verts, indices=self.edge[:, 1], axis=1)
        new_vert = (verts_0 + verts_1) / 2

        return tf.concat([verts, new_vert], axis=1, name="mesh_x_concat")

    def __call__(self, beta, theta, trans=None, v_personal=None, other=None, name=None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """

        with tf.name_scope(name, "smpl_main", [beta, theta]):
            num_batch = beta.shape[0].value
            nums = beta.shape[1].value

            # 1. Add shape blend shapes
            # (N x 10) x (10 x 6890*3) = N x 6890 x 3
            # print(num_batch)
            # exit()

            v_shaped = tf.reshape(
                tf.matmul(beta, self.shapedirs, name='shape_bs'),
                [-1, self.size[0], self.size[1]]) + self.v_template
            # v_shaped = beta
            
            if v_personal is not None:
                v_shaped_personal = self.mesh_x(v_shaped) + v_personal
                # v_shaped_personal = v_shaped + v_personal
            else:
                v_shaped_personal = v_shaped
            # 2. Infer shape-dependent joint locations.
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor)
            J = tf.stack([Jx, Jy, Jz], axis=2)   #[16,24,3]


            theta = self.masktf*theta
            # 3. Add pose blend shapes
            # N x 24 x 3 x 3
            Rs = tf.reshape(
                batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
            with tf.name_scope("lrotmin"):
                # Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3),
                                          [-1, 207])

            # (N x 207) x (207, 20670) -> N x 6890 x 3
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)
            if v_personal is not None:
                v_posed = tf.reshape(
                    tf.matmul(pose_feature, self.posedirs),
                    [-1, self.size[0], self.size[1]]) 
                v_posed = self.mesh_x(v_posed) + v_shaped_personal
                # v_posed += v_shaped_personal

                W = tf.reshape(
                    tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, 24])

                W = self.mesh_x(W)
                T = tf.reshape(
                    tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
                    [num_batch, -1, 4, 4])
                # T = self.mesh_x(T)

            else:
                v_posed = tf.reshape(
                    tf.matmul(pose_feature, self.posedirs),
                    [-1, self.size[0], self.size[1]]) + v_shaped

                W = tf.reshape(
                    tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, 24])
                T = tf.reshape(
                    tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
                    [num_batch, -1, 4, 4])

            v_posed_homo = tf.concat(
                [v_posed, tf.ones([num_batch, v_posed.shape[1], 1])], 2)
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

            verts = v_homo[:, :, :3, 0]

            if trans is not None:

                verts += trans

            return verts, v_shaped_personal




if __name__ == "__main__":
    from write2obj import write_to_obj
    s = SMPL()
    beta = tf.placeholder(tf.float32, [1, 10])
    theta = tf.placeholder(tf.float32, [1, 72])
    trans = tf.placeholder(tf.float32, [1, 3])
    offset = tf.placeholder(tf.float32, [1, 16535, 3])
    pred_vert, _ = s(beta, theta, trans, offset)



    with tf.Session() as sess:
        all_theta = np.loadtxt("../fittingcode_ourman2/cape/example.txt") #TODO:
        all_theta = np.reshape(all_theta, [-1, 85])
        off = np.loadtxt('test/cape_male/pred_offset_0.txt')
        verts = sess.run(
            pred_vert, 
            {
                theta : all_theta[:, :72], 
                beta : all_theta[:, 72:82],
                trans : all_theta[:, 82:],
                offset : off[None, ...]
            }
        )

        write_to_obj('test/cape_male/repose_1.obj', verts)

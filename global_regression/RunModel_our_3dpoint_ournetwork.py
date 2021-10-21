import tensorflow as tf
import numpy as np
from os.path import exists

# from tf_smpl import projection as proj_util
from tf_smpl.batch_smpl_our import SMPL
from models import get_encoder_fn_separate, Encoder_point
from tf_smpl.batch_lbs import batch_rodrigues

import pointnet
import pointnet2
class RunModel(object):
    def __init__(self, config, sess=None):
        """
        Args:
          config
        """
        self.config = config
        self.load_path = config.load_path
        
        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You need to specify `load_path` to load a pretrained model"
            )
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            # import ipdb
            # ipdb.set_trace()

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        
        #input_size = (self.batch_size, self.img_size, self.img_size, 1)
        input_size = (1, 2500, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)

        # Model Settings
        self.num_stage = config.num_stage
        self.model_type = config.model_type

        # Camera
        self.num_cam = 3
        # self.proj_fn = proj_util.batch_orth_proj_idrot

        self.num_theta = 72        
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10

        self.smpl = SMPL(self.smpl_model_path)

        self.build_test_model_ief()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        # Load data.
        self.saver = tf.train.Saver()
        self.prepare()        

    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        #mean[0] = np.pi
        mean_path = './meanpara_male.txt'
        mean_vals = np.loadtxt(mean_path, dtype=np.float32)

        mean_pose = mean_vals[:72]
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals[72:]

        #This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, :self.total_params-3] = np.hstack((mean_pose, mean_shape))

        mean = tf.constant(mean, tf.float32)
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=False)
        #self.E_var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return init_mean


    def build_test_model_ief(self):

        def rot6d_to_rotmat(x):
            """Convert 6D rotation representation to 3x3 rotation matrix.
            Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
            Input:
                (B,6) Batch of 6-D rotation representations
            Output:
                (B,3,3) Batch of corresponding rotation matrices
            """
            x = tf.reshape(x, [-1,3,2])
            a1 = x[:, :, 0]
            a2 = x[:, :, 1]
            b1 = tf.nn.l2_normalize(a1,dim=1)
            b2 = tf.nn.l2_normalize(a2 - tf.expand_dims(tf.einsum('bi,bi->b', b1, a2),-1) * b1, dim=1)
            b3 = tf.cross(b1, b2)
            return tf.concat([b1, b2, b3], 1)

        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)

        inputpoints = self.images_pl  # self.kp_loader[:,:,0:3]
        self.img_feat, self.E_var= pointnet2.extract_globalfeature(inputpoints, None, tf.cast(False, tf.bool))


        pred_pose, pred_shape, pred_cam = self.load_mean_param1()

        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.final_thetas = []

        # Main IEF loop
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, pred_pose, pred_shape, pred_cam], 1)

            if i == 0:
                delta_pose, delta_shape, delta_cam, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_pose, delta_shape, delta_cam, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)


            pred_pose = pred_pose + delta_pose
            pred_shape = pred_shape + delta_shape
            pred_cam = pred_cam + delta_cam
            pred_rotmat = tf.reshape(rot6d_to_rotmat(pred_pose), [self.batch_size, 24, 3, 3])
            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            verts, pred_Rs = self.smpl(pred_shape, pred_rotmat)

            pred_kp = verts + tf.reshape(tf.tile(pred_cam,[1,6890]),[-1,6890,3])#tranlating
            theta_here = tf.concat([pred_pose, pred_shape, pred_cam], 1)
            self.all_verts.append(pred_kp)
            self.final_thetas.append(theta_here)
            # Finally update to end iteration.

    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)        
        #self.mean_value = self.sess.run(self.mean_var)

    def predict_dict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        feed_dict = {
            self.images_pl: images,
        }
        fetch_dict = {
            'verts': self.all_verts[-1],
            'theta': self.final_thetas[-1],
        }

        results = self.sess.run(fetch_dict, feed_dict)
        # print(results['loss'])

        return results['theta'], results['verts']

    def savemodel(self, verts):

        v_triangle = self.smpl.v_face
        v_trianglesize = np.shape(v_triangle)[0]

        verts = np.squeeze(verts, 0)#self.smpl.verts
        pname = './tmodel.obj'#
        f = open(pname, "w+")

        for i in range(np.shape(verts)[0]):
            f.write("v " + str(verts[i,0])+ " "+ str(verts[i,1])+ " "+ str(verts[i,2]))
            f.write("\n")

        for i in range(v_trianglesize):
            f.write("f " + str(v_triangle[i,0]+1)+ " "+ str(v_triangle[i,1]+1)+ " "+ str(v_triangle[i,2]+1))
            f.write("\n")
        f.close()
    
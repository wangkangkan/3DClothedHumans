""" Evaluates a trained model using placeholders. """

import tensorflow as tf
import numpy as np
from os.path import exists

from tf_smpl import projection as proj_util
from tf_smpl.batch_smpl_our import SMPL
from ops import keypoint3D_loss
from models import get_encoder_fn_separate
from tf_smpl.batch_lbs import batch_rodrigues

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
        # self.initpara = tf.placeholder(tf.float32, shape=(1,85))
        self.initpara = tf.placeholder(tf.float32, shape=(1,24*6+10+3))
        self.kp_loader = tf.placeholder(tf.float32, shape=(1, 6890, 4))


        npmodelpoint = np.loadtxt('./smplzero.txt', dtype=np.float32)
        npmodelpoint = np.reshape(npmodelpoint, [1, 6890, 3])
        modelpoint = tf.constant(npmodelpoint, tf.float32)
        self.batchmodelpoint = tf.tile(modelpoint, [self.batch_size, 1, 1])

        # Model Settings
        self.num_stage = config.num_stage
        self.model_type = config.model_type
        self.keypoint_loss = keypoint3D_loss

        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd
        self.e_loss_weight = config.e_loss_weight
        self.d_loss_weight = config.d_loss_weight
        self.e_3d_weight = config.e_3d_weight

        # Camera
        self.num_cam = 3
        self.num_theta = 24*6        
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10

        self.smpl = SMPL(self.smpl_model_path)

        self.p_idx = np.loadtxt('108idx.txt',dtype=np.int32)
        self.build_test_model_ief()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        # Load data.
        self.saver = tf.train.Saver()
        self.prepare()


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

        gdpara = self.initpara
        pred_pose = gdpara[:, :self.num_theta]
        pred_shape = gdpara[:, self.num_theta:self.num_theta + 10]
        pred_cam = gdpara[:, self.num_theta + 10:self.num_theta + 10 + 3]
        print(pred_pose, self.num_theta)
        pred_rotmat = tf.reshape(rot6d_to_rotmat(pred_pose), [self.batch_size, 24, 3, 3])
        print(pred_shape, pred_rotmat)
        initverts, _ = self.smpl(pred_shape, pred_rotmat)
        self.initverts = initverts + tf.reshape(tf.tile(pred_cam, [1, 6890]), [-1, 6890, 3])

        point = tf.transpose(self.initverts,[1,0,2])
        self.point = tf.transpose(tf.gather(point,self.p_idx),[1,0,2])
        inputpoints = self.images_pl
        self.img_feat, self.E_var= pointnet2.extract_jointsfeature(inputpoints, None, self.point, is_training=False, bn_decay=None)

        atten_imfeat = tf.reshape(self.img_feat, [self.batch_size, -1])
        loss_kps = []


        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.final_thetas = []

        # Main IEF loop
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([atten_imfeat, pred_pose, pred_shape, pred_cam], 1)

            if i == 0:
                delta_pose, delta_shape, delta_cam, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_pose, delta_shape, delta_cam, _ = threed_enc_fn(
                    state, num_output=self.total_params, is_training=False, reuse=True)

            # Compute new theta
            pred_pose = pred_pose + delta_pose
            pred_shape = pred_shape + delta_shape
            pred_cam = pred_cam + delta_cam
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            pred_rotmat = tf.reshape(rot6d_to_rotmat(pred_pose), [self.batch_size, 24, 3, 3])

            verts, pred_Rs = self.smpl(pred_shape, pred_rotmat)

            pred_kp = verts + tf.reshape(tf.tile(pred_cam,[1,6890]),[-1,6890,3])#tranlating
            theta_here = tf.concat([pred_pose, pred_shape, pred_cam], 1)
            loss_kps.append(self.keypoint_loss(
                self.kp_loader, pred_kp))
            
            self.all_verts.append(pred_kp)
            self.final_thetas.append(theta_here)


        self.e_loss_kp = loss_kps[-1]

    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)        
        #self.mean_value = self.sess.run(self.mean_var)

    def predict(self, images, get_theta=False):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        """
        results = self.predict_dict(images)
        if get_theta:
            return results['verts'], results['theta']
        else:
            return results['verts']

    def predict_dict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        feed_dict = {
            self.images_pl: images,
            # self.theta0_pl: self.mean_var,
        }
        fetch_dict = {
            'verts': self.all_verts[-1],
            'theta': self.final_thetas[-1],
        }

        results = self.sess.run(fetch_dict, feed_dict)

        return results

    def predict_T(self, images, initpara):
    
        theta = self.final_thetas[-1]

        cams_trl = theta[0, self.num_theta + 10:self.num_theta + 10 + 3]


        feed_dict = {
            self.images_pl: images,
            self.initpara: initpara,
        }
        fetch_dict = {
            'verts': self.all_verts[-1],
            'theta': theta,
            'cams_trl': cams_trl,
        }

        results = self.sess.run(fetch_dict, feed_dict)

        return results['verts'], results['theta'], results['cams_trl']
        
    def savemodel_T(self, verts, cams_trl, idx):
    
        v_triangle = self.smpl.v_face
        v_trianglesize = np.shape(v_triangle)[0]

        finalverts = verts 

        pname = './result/tmodel_{0}.obj'.format(idx) 
        f = open(pname, "w+")

        for i in range(np.shape(finalverts)[0]):
            f.write("v " + str(finalverts[i, 0]) + " " + str(finalverts[i, 1]) + " " + str(finalverts[i, 2]))
            f.write("\n")

        for i in range(v_trianglesize):
            f.write("f " + str(v_triangle[i, 0] + 1) + " " + str(v_triangle[i, 1] + 1) + " " + str(
                v_triangle[i, 2] + 1))
            f.write("\n")
        f.close()
   
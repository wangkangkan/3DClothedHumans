from data_loader_our_3dpoint import num_examples

from ops import compute_3d_loss, compute_3d_loss_our, align_by_pelvis, keypoint3D_loss
from models import Discriminator_separable_rotations, get_encoder_fn_separate

from tf_smpl.batch_lbs import batch_rodrigues
from tf_smpl.batch_smpl_our import SMPL
from tf_smpl.projection import batch_orth_proj_idrot

from tensorflow.python.ops import control_flow_ops

from time import time
import tensorflow as tf
import numpy as np


import pointnet2

class Trainer(object):
    def __init__(self, config, data_loader, mocap_loader):
        """
        Args:
          config
          if no 3D label is available,
              data_loader is a dict
          else
              data_loader is a dict
        mocap_loader is a tuple (pose, shape)
        """
        # Config + path
        self.config = config
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.pretrained_model_path = config.pretrained_model_path
        self.encoder_only = config.encoder_only
        self.use_3d_label = config.use_3d_label

        # Data size
        self.img_size = config.img_size
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size
        self.max_epoch = config.epoch

        self.num_cam = 3#translation

        #changed to RT
        self.proj_fn = batch_orth_proj_idrot

        #including R
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        # Data
        num_images = num_examples(config.datasets)
        num_mocap = num_examples(config.mocap_datasets)

        self.num_itr_per_epoch = int(num_images / self.batch_size)
        self.num_mocap_itr_per_epoch = num_mocap / self.batch_size

        self.pointset_loader = data_loader['pointset']
        self.kp_loader = data_loader['label']#3D point locations
        self.poseshape_loader = data_loader['para']
        self.pose_loader = mocap_loader[0]
        self.shape_loader = mocap_loader[1]

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.log_img_step = config.log_img_step

        # Model spec
        self.model_type = config.model_type

        # Optimizer, learning rate
        self.e_lr = config.e_lr
        self.d_lr = config.d_lr
        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd
        self.e_loss_weight = config.e_loss_weight
        self.d_loss_weight = config.d_loss_weight
        self.e_3d_weight = config.e_3d_weight

        self.optimizer = tf.train.AdamOptimizer
        self.keypoint_loss = keypoint3D_loss
        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)
        self.E_var = []
        self.build_model()

        # Logging
        init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            print("Fine-tuning from %s" % self.pretrained_model_path)
            if 'resnet_v2_50' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v2_50' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            elif 'pose-tensorflow' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v1_101' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            else:
                self.pre_train_saver = tf.train.Saver()

            def load_pretrain(sess):
                self.pre_train_saver.restore(sess, self.pretrained_model_path)

            init_fn = load_pretrain

        self.saver = tf.train.Saver(max_to_keep=10000, keep_checkpoint_every_n_hours=5)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            global_step=self.global_step,
            saver=self.saver,
            summary_writer=self.summary_writer,
            init_fn=init_fn)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_options)

    def use_pretrained(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty, meaning we're picking up from previous
             so fuck this pretrained model.
        """
        if ('resnet' in self.model_type) and (self.pretrained_model_path is
                                              not None):
            # Check is model_dir is empty
            import os
            if os.listdir(self.model_dir) == []:
                return True

        return False

    def load_mean_param(self):
        # mean = np.zeros((1, self.total_params))
        # #mean[0] = np.pi
        # mean_path = './meanpara_male.txt'
        # mean_vals = np.loadtxt(mean_path, dtype=np.float32)

        # mean_pose = mean_vals[:72]
        # # Ignore the global rotation.
        # mean_pose[:3] = 0.
        # mean_shape = mean_vals[72:]

        # #This initializes the global pose to be up-right when projected
        # mean_pose[0] = np.pi

        # mean[0, :self.total_params-3] = np.hstack((mean_pose, mean_shape))

        # mean = tf.constant(mean, tf.float32)
        # self.mean_var = tf.Variable(
        #     mean, name="mean_param", dtype=tf.float32, trainable=True)
        #self.E_var.append(self.mean_var)
        
        mean_params = np.load('smpl_mean_params.npz')
        init_pose = mean_params['pose'][:].reshape(1,-1)
        init_shape = mean_params['shape'][:].reshape(1,-1)
        init_cam = mean_params['cam'].reshape(1,-1)
        self.init_pose = tf.Variable(
            init_pose, name="init_pose", dtype=tf.float32, trainable=True)
        self.init_shape = tf.Variable(
            init_shape, name="init_shape", dtype=tf.float32, trainable=True)
        self.init_cam = tf.Variable(
            init_cam, name="init_cam", dtype=tf.float32, trainable=True)
        # mean = np.hstack((init_pose, init_pose))
        # self.mean_var = tf.Variable(
        #     mean, name="mean_param", dtype=tf.float32, trainable=True)
        init_pose = tf.tile(self.init_pose, [self.batch_size, 1])
        init_shape = tf.tile(self.init_shape, [self.batch_size, 1])
        init_cam = tf.tile(self.init_cam, [self.batch_size, 1])
        return init_pose, init_shape, init_cam

    def build_model(self):

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
  
        self.img_feat, self.E_var= pointnet2.extract_globalfeature(self.pointset_loader, None, tf.cast(True, tf.bool))

        loss_kps = []
        loss_3d_params = []
        fake_rotations, fake_shapes = [], []

        pred_pose, pred_shape, pred_cam = self.load_mean_param()
        self.E_var.extend([self.init_pose, self.init_shape, self.init_cam])

        # For visualizations
        self.all_verts = []
        self.all_pred_kps = []
        self.all_pred_cams = []
        self.all_delta_thetas = []
        self.all_theta_prev = []

        # Main IEF loop
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, pred_pose, pred_shape, pred_cam], 1)

            if i == 0:
                delta_pose, delta_shape, delta_cam, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_pose, delta_shape, delta_cam, _ = threed_enc_fn(
                    state, num_output=self.total_params, reuse=True)

            pred_pose = pred_pose + delta_pose
            pred_shape = pred_shape + delta_shape
            pred_cam = pred_cam + delta_cam
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            pred_rotmat = tf.reshape(rot6d_to_rotmat(pred_pose), [self.batch_size, 24, 3, 3])
            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            verts, pred_Rs = self.smpl(pred_shape, pred_rotmat)

            pred_kp = verts + tf.reshape(tf.tile(pred_cam,[1,6890]),[-1,6890,3])#tranlating

            # --- Compute losses:
            loss_kps.append(self.e_loss_weight * self.keypoint_loss(
                self.kp_loader, pred_kp))


            pred_Rs = tf.reshape(pred_Rs, [-1, 24, 9])

            loss_poseshaperottransl = self.get_3d_loss_our_withoutRT(
                pred_Rs, pred_shape,pred_cam)
            loss_3d_params.append(loss_poseshaperottransl)

            # Save pred_rotations for Discriminator
            fake_rotations.append(pred_Rs[:, 1:, :])
            fake_shapes.append(pred_shape)

            # Finally update to end iteration.
            # theta_prev = theta_here

        if not self.encoder_only:
            self.setup_discriminator(fake_rotations, fake_shapes)

        # Gather losses.
        with tf.name_scope("gather_e_loss"):
            # Just the last loss.
            self.e_loss_kp = loss_kps[-1] + loss_3d_params[-1]

            if self.encoder_only:
                self.e_loss = self.e_loss_kp
            else:
                self.e_loss = self.d_loss_weight * self.e_loss_disc + self.e_loss_kp

            self.e_loss_3d = tf.constant(0)
        if not self.encoder_only:
            with tf.name_scope("gather_d_loss"):
                self.d_loss = self.d_loss_weight * (
                    self.d_loss_real + self.d_loss_fake)

        # Don't forget to update batchnorm's moving means.
        print('collecting batch norm moving means!!')
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.e_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.e_loss)

        # Setup optimizer
        print('Setting up optimizer..')
        d_optimizer = self.optimizer(self.d_lr)
        e_optimizer = self.optimizer(self.e_lr)

        self.e_opt = e_optimizer.minimize(
            self.e_loss, global_step=self.global_step, var_list=self.E_var)
        if not self.encoder_only:
            self.d_opt = d_optimizer.minimize(self.d_loss, var_list=self.D_var)

        print('Done initializing trainer!')

    def setup_discriminator(self, fake_rotations, fake_shapes):
        # Compute the rotation matrices of "rea" pose.
        # These guys are in 24 x 3.
        real_rotations = batch_rodrigues(tf.reshape(self.pose_loader, [-1, 3]))
        real_rotations = tf.reshape(real_rotations, [-1, 24, 9])
        # Ignoring global rotation. N x 23*9
        # The # of real rotation is B*num_stage so it's balanced.
        real_rotations = real_rotations[:, 1:, :]
        all_fake_rotations = tf.reshape(
            tf.concat(fake_rotations, 0),
            [self.batch_size * self.num_stage, -1, 9])
        comb_rotations = tf.concat(
            [real_rotations, all_fake_rotations], 0, name="combined_pose")

        comb_rotations = tf.expand_dims(comb_rotations, 2)
        all_fake_shapes = tf.concat(fake_shapes, 0)
        comb_shapes = tf.concat(
            [self.shape_loader, all_fake_shapes], 0, name="combined_shape")

        disc_input = {
            'weight_decay': self.d_wd,
            'shapes': comb_shapes,
            'poses': comb_rotations
        }

        self.d_out, self.D_var = Discriminator_separable_rotations(
            **disc_input)

        self.d_out_real, self.d_out_fake = tf.split(self.d_out, 2)
        # Compute losses:
        with tf.name_scope("comp_d_loss"):
            self.d_loss_real = tf.reduce_mean(
                tf.reduce_sum((self.d_out_real - 1)**2, axis=1))
            self.d_loss_fake = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake)**2, axis=1))
            # Encoder loss
            self.e_loss_disc = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake - 1)**2, axis=1))

    def get_3d_loss_our_withoutRT(self, Rs, shape, translation):
        """
        Rs is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        Js is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        tRs = tf.reshape(Rs[:, 1:, :], [self.batch_size, -1])
        params_pred = tf.concat([tRs, shape], 1, name="prep_params_pred")
        # 24*9+10 = 226
        gt_params = self.poseshapetranslation_loader[:, 9:226]
        loss_poseshapetranslation = self.e_3d_weight * compute_3d_loss_our(
            params_pred, gt_params)

        return loss_poseshapetranslation

    def get_3d_loss(self, Rs, shape, Js):
        """
        Rs is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        Js is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        Rs = tf.reshape(Rs, [self.batch_size, -1])
        params_pred = tf.concat([Rs, shape], 1, name="prep_params_pred")
        # 24*9+10 = 226
        gt_params = self.poseshape_loader[:, :226]
        loss_poseshape = self.e_3d_weight * compute_3d_loss(
            params_pred, gt_params, self.has_gt3d_smpl)
        # 14*3 = 42
        gt_joints = self.poseshape_loader[:, 226:]
        pred_joints = Js[:, :14, :]
        # Align the joints by pelvis.
        pred_joints = align_by_pelvis(pred_joints)
        pred_joints = tf.reshape(pred_joints, [self.batch_size, -1])
        gt_joints = tf.reshape(gt_joints, [self.batch_size, 14, 3])
        gt_joints = align_by_pelvis(gt_joints)
        gt_joints = tf.reshape(gt_joints, [self.batch_size, -1])

        loss_joints = self.e_3d_weight * compute_3d_loss(
            pred_joints, gt_joints, self.has_gt3d_joints)

        return loss_poseshape, loss_joints

    def train(self):
        step = 0

        epoche_loss = 0
        epochd_loss = 0
        epochk_loss = 0
        epochloss_3d = 0
        iter = 0

        with self.sv.managed_session(config=self.sess_config) as sess:
            while not self.sv.should_stop():
                fetch_dict = {
                    "step": self.global_step,
                    "e_loss": self.e_loss,
                    # The meat
                    "e_opt": self.e_opt,
                    "loss_kp": self.e_loss_kp,
                    #"feat": self.img_feat
                }
                if not self.encoder_only:
                    fetch_dict.update({
                        # For D:
                        "d_opt": self.d_opt,
                        "d_loss": self.d_loss,
                        "loss_disc": self.e_loss_disc,
                    })
                fetch_dict.update({
                    "loss_3d_params": self.e_loss_3d
                })
            
                t0 = time()
                result = sess.run(fetch_dict)
                t1 = time()

                e_loss = result['e_loss']
                step = result['step']

                epoch = float(step) / self.num_itr_per_epoch
                if self.encoder_only:
                    print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f" %
                          (step, epoch, t1 - t0, e_loss))
                else:
                    d_loss = result['d_loss']
                    loss_kp = result['loss_kp']
                    loss_3d = result['loss_3d_params']
                    print(
                        "itr %d/(epoch %.1f): time %g, Enc_loss: %.4f, Disc_loss: %.4f, kp_loss: %.4f, loss_3d_params: %.4f"
                        % (step, epoch, t1 - t0, e_loss, d_loss, loss_kp, loss_3d))

                epoche_loss = epoche_loss + e_loss
                epochd_loss = epochd_loss + d_loss
                epochk_loss = epochk_loss + loss_kp
                epochloss_3d = epochloss_3d+ loss_3d
                iter = iter + 1
                if step % self.num_itr_per_epoch == 0 and step >= 1:
                    # self.saver.save(sess,'./logs/models/model.ckpt', global_step = int(epoch))
                    f = open('./loss0807.txt', 'a+')
                    epoche_loss = epoche_loss / iter
                    epochd_loss = epochd_loss / iter
                    epochk_loss = epochk_loss / iter
                    epochloss_3d = epochloss_3d / iter
                    f.write(str("%d %.5f %.5f %.5f %.5f" % (epoch, epoche_loss, epochd_loss, epochk_loss, epochloss_3d)))
                    f.write('\n')
                    f.close()
                    epoche_loss = 0
                    epochd_loss = 0
                    epochk_loss = 0
                    epochloss_3d = 0
                    iter = 0

                if epoch > self.max_epoch:
                    self.sv.request_stop()

        print('Finish training on %s' % self.model_dir)

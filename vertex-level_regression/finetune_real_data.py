'''
    Single-GPU training code
'''

import argparse
import glob
import importlib
import os
import sys

import natsort
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import time

# from old_evaluate import savemodel
from tf_smpl.batch_smpl_our_real_data import SMPL

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='../flownet3d/human_1723_3000_stage1/', help='Dataset directory [default: data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--log_dir', default='log_train_realdata', help='Log dir [default: log_train]')
parser.add_argument('--num_point', type=int, default=6890, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=52, help='Epoch to run [default: 151]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=40000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=1.0, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--is_training', type=bool, default=True, help='is_train [default: True]')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
LABEL_CHANNEL = 3
EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
IS_TRAIN = FLAGS.is_training
POINT_NUMBER = 20000

MODEL_PATH = './log_train_cape_stage1_20w_predict/model.ckpt'

SMPL_MODEL_PATH = './tf_smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'

NUM_THETA = 72
NUM_SHAPE = 10

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_DATASET = 30000

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def mesh_x(verts, edge):
    verts_0 = tf.gather(verts, indices=edge[:, 0], axis=1)
    verts_1 = tf.gather(verts, indices=edge[:, 1], axis=1)
    new_vert = (verts_0 + verts_1) / 2

    return tf.concat([verts, new_vert], axis=1, name="mesh_x")


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(train_flag=True):
    # globalstep = 0
    if train_flag:
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(GPU_INDEX)):
                pointclouds_pl = {}

                labels_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 85])
                pointclouds_pl['pointcloud'] = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_POINT+POINT_NUMBER, 6])
                is_training_pl = tf.placeholder(dtype=tf.bool)

                smpl = SMPL(SMPL_MODEL_PATH)


                #################for generate 16535 points mesh ######
                edge_for_16535 = np.loadtxt('edge_16535.txt', dtype=np.int)
                edge_for_16535_tf = tf.constant(edge_for_16535, dtype=tf.int32, shape=[9645, 2])

                ##### edge loss and laplacian loss############
                conname = 'laplacian_connect.txt'
                con = np.loadtxt(conname, dtype=np.int)
                edgename = 'edge_for_loss_6890.txt'
                edge = np.loadtxt(edgename, dtype=np.int)

                face = np.loadtxt('./normal_face_6890_x6.txt', dtype=np.int) - 1
                face_for_normal = tf.constant(face, dtype=tf.int32)
                ##### 108 joint point ######
                p_idx = np.loadtxt('./108idx.txt',dtype=np.int32)
                IDX_JOINT = tf.constant(p_idx, dtype=tf.int32)


                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)

                ##### SMPL ######
                pose = labels_pl[:, :NUM_THETA]
                shape = labels_pl[:, NUM_THETA : NUM_THETA + NUM_SHAPE]
                trans = tf.tile(tf.reshape(labels_pl[:, NUM_THETA + NUM_SHAPE:], [BATCH_SIZE, 1, 3]), [1, 16535, 1])

                pred_verts_smpl = pointclouds_pl['pointcloud'][:, :NUM_POINT, :3]
                
                input_pc = pointclouds_pl['pointcloud'][:, NUM_POINT:, :3]
                normal = pointclouds_pl['pointcloud'][:, NUM_POINT:, 3:6]

                joint_point = tf.gather(pred_verts_smpl, indices=IDX_JOINT, axis=1)

                pointclouds_pl['joint_point'] = joint_point

                print("--- Get model and loss")
                # Get model and loss
                pred, all_data = MODEL.get_model(pointclouds_pl, 
                                                    is_training_pl, 
                                                    bn_decay=bn_decay)
                pred_verts, pred_Rs = smpl(shape, pose, trans=trans, v_personal=pred)

                ############# cal normal ########################
                meshnormal = MODEL.cal_normals(pred_verts[:, :6890, :], face_for_normal)

                ############## cal loss #########################################
                loss, pre_vert_20000, idx = MODEL.nearest_loss(input_pc, pred_verts[:, :6890, :], normal, meshnormal)
                loss = 10*loss #+ 10*loss2
                
                loss_laplacian = MODEL.Laplacian_loss(pred_verts_smpl, pred_verts[:, :NUM_POINT, :], con)
                loss_edge = MODEL.edge_loss(pred_verts_smpl, pred_verts[:, :NUM_POINT, :], edge)

                ############### hand offset loss ######################
                mask_for_classfication = np.loadtxt('idx_for_head_hand_foot.txt', dtype=int)
                mask_for_classfication_tf = tf.constant(mask_for_classfication, name='mask_for_classfication', dtype=tf.int32)
                
                loss_offset_hand_foot = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.gather(pred, indices=mask_for_classfication_tf, axis=1)), axis=[1, 2]))

                loss += 20*loss_offset_hand_foot


                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):

                    train_op = optimizer.minimize(loss + 0.1*(loss_edge + loss_laplacian), global_step=batch)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            coord = tf.train.Coordinator() 
            threads = tf.train.start_queue_runners(sess, coord)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)
            


            if os.path.exists(MODEL_PATH+'.index'):
                saver.restore(sess, MODEL_PATH)
                print('\n!!!!!!!!!!!!!!restore from ', MODEL_PATH)


            ops = {'is_training_pl': is_training_pl,
                'pl': pointclouds_pl['pointcloud'],
                'labels': labels_pl,
                'pred': pred_verts,
                'pred_smpl': pred_verts_smpl,
                'loss': loss,
                'learning_rate': learning_rate, 
                'loss_laplacian': loss_laplacian,
                'loss_edge': loss_edge,
                'train_op': train_op,
                'step': batch,
                }

            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                loss_average, true_epoch = train_one_epoch(sess, ops, saver)
                # Save the variables to disk.
                if epoch % 1 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)
            
            coord.request_stop()
            coord.join(threads)
   
def train_one_epoch(sess, ops, saver):
    """ ops: dict mapping from string to tf ops """

    is_training = IS_TRAIN

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT+POINT_NUMBER, 6))
    batch_label = np.ones((BATCH_SIZE, 85))

    global_count = 0
    global_count2 = 0

    data_cate = 'data3'

    #### raw_obj
    real_pointcloud_3000 = natsort.natsorted(glob.glob('../displacement/{}/6890/*.txt'.format(data_cate)))

    real_pointcloud_10000 = natsort.natsorted(glob.glob('../displacement/{}/{:d}/*.txt'.format(data_cate, POINT_NUMBER)))

    #### 85-D theta
    all_theta = np.loadtxt("../fittingcode_ourman2/ismar/theta.txt").reshape((1, -1)) #TODO:
    all_theta = np.tile(all_theta, [BATCH_SIZE*len(real_pointcloud_3000), 1])
    
    #### 10000-normal
    real_pointcloud_normal = natsort.natsorted(glob.glob("../displacement/{}/normal_{:d}/*normal.txt".format(data_cate, POINT_NUMBER)))

    all_idx_random = np.random.permutation(len(real_pointcloud_3000))

    num_batches = len(real_pointcloud_3000) // BATCH_SIZE*10
    keyword = ''
    path = './test/{}'.format(keyword)


    loss_sum = 0
    loss_batch = 0

    for batch_idx in range(0, num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        xyz2list = all_idx_random[start_idx:end_idx]

        for i, line in enumerate(xyz2list):
            batch_data[i, :NUM_POINT, :3] = np.loadtxt(real_pointcloud_3000[line])
            batch_data[i, NUM_POINT:, :3] = np.loadtxt(real_pointcloud_10000[line])
            batch_data[i, NUM_POINT:, 3:6] = np.loadtxt(real_pointcloud_normal[line])
            batch_label[i]  = all_theta[line]
            global_count2 += 1

        feed_dict = {
                     ops['is_training_pl']: is_training,
                    #  ops['classes']: load_class,
                     ops['pl']: batch_data,
                     ops['labels']: batch_label,
                     }
        timestart = time.time()
        step, _, loss_val, learning_rate_pl, loss_laplacian_val, loss_edge_val = sess.run([ops['step'], \
                                                                            ops['train_op'], \
                                                                            ops['loss'], \
                                                                            ops['learning_rate'], \
                                                                            ops['loss_laplacian'], \
                                                                            ops['loss_edge']], \
                                                                            feed_dict)
        timeend = time.time()
        loss_sum += loss_val
        loss_batch += loss_val
        print('time::%.3f, lr:%.4f, step:%d, loss: %.8f, loss_lp: %.03f, loss_edge: %.3f' % \
            ((timeend-timestart), learning_rate_pl, step, loss_val, loss_laplacian_val, loss_edge_val))


    return loss_batch / num_batches, step//num_batches



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train_flag = True
    train(train_flag)
    LOG_FOUT.close()

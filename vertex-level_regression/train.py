'''
    Single-GPU training code
'''

import argparse
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import natsort
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import time


from tf_smpl.batch_smpl_our_predict import SMPL


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='../displacement/data_20w/tfrecord_stage1_predict/', help='Dataset directory')
parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')
parser.add_argument('--num_point', type=int, default=6890, help='Point Number [default: 6890]')
parser.add_argument('--max_epoch', type=int, default=52, help='Epoch to run [default: 52]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=40000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.9]')
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
POINT_NUMBER = 20000 # input pointcloud point number

MODEL_PATH = './log_train/model.ckpt'

SMPL_MODEL_PATH = './tf_smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'

NUM_THETA = 72*2 # pose 
NUM_SHAPE = 10 # shape

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

TRAIN_DATASET = 200000

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

def read_file_list(filelist):
    """
    Scan the image file and get the image paths and labels
    """
    with open(filelist) as f:
        lines = f.readlines()
        files = []
        for l in lines:
            items = l.split()
            files.append(items[0])
            #self.imagefiles.append(l)

        # store total number of data
    filenum = len(files)
    print("Training sample number: %d" % (filenum))
    return files

def parse_example(example_serialized, pointnum=6890, pointnum2=POINT_NUMBER):
    """Parses an Example proto."""
    feature_map = {
        'frame1/x': tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'frame1/y': tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'frame1/z': tf.FixedLenFeature((pointnum, 1), dtype=tf.float32),
        'frame2/x': tf.FixedLenFeature((pointnum2, 1), dtype=tf.float32),
        'frame2/y': tf.FixedLenFeature((pointnum2, 1), dtype=tf.float32),
        'frame2/z': tf.FixedLenFeature((pointnum2, 1), dtype=tf.float32),
        'flow/x' : tf.FixedLenFeature((160, 1), dtype=tf.float32),
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    frame1_x = tf.cast(features['frame1/x'], dtype=tf.float32)
    frame1_y = tf.cast(features['frame1/y'], dtype=tf.float32)
    frame1_z = tf.cast(features['frame1/z'], dtype=tf.float32)
    frame1_x = tf.reshape(frame1_x, [pointnum, 1])
    frame1_y = tf.reshape(frame1_y, [pointnum, 1])
    frame1_z = tf.reshape(frame1_z, [pointnum, 1])
    frame1 = tf.concat([frame1_x, frame1_y, frame1_z], axis=1)

    feat1 = tf.zeros([pointnum, 101])
    feat2 = tf.zeros([pointnum2, 101])

    frame2_x = tf.cast(features['frame2/x'], dtype=tf.float32)
    frame2_y = tf.cast(features['frame2/y'], dtype=tf.float32)
    frame2_z = tf.cast(features['frame2/z'], dtype=tf.float32)
    frame2_x = tf.reshape(frame2_x, [pointnum2, 1])
    frame2_y = tf.reshape(frame2_y, [pointnum2, 1])
    frame2_z = tf.reshape(frame2_z, [pointnum2, 1])
    frame2 = tf.concat([frame2_x, frame2_y, frame2_z], axis=1)

    frame = tf.concat([frame1, frame2], axis=0)
    color = tf.concat([feat1, feat2], axis=0)
    frame = tf.concat([frame, color], axis=1)

    flow_x = tf.cast(features['flow/x'], dtype=tf.float32)
    flow = tf.reshape(flow_x, [160])

    return frame, flow

def get_batch(batch_size, fqueue):

    with tf.name_scope(None, 'read_data', [fqueue]):
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(fqueue)
        frame, flow = parse_example(example_serialized, pointnum=NUM_POINT)
        min_after_dequeue = 160
        num_threads = 4
        capacity = min_after_dequeue + 30 * batch_size

        pack_these = [frame, flow]
        pack_name = ['frame', 'flow']
        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=False,
            name='input_batch_train')
        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict

def train(train_flag=True):
    # globalstep = 0
    if train_flag:
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(GPU_INDEX)):

                data_dirs = os.listdir(DATA)
 
                all_files = []
                for data_dir in data_dirs:
                    all_files.append(DATA + data_dir)

                pointclouds_pl = {}
                do_shuffle = True
                fqueue = tf.train.string_input_producer(all_files, shuffle=do_shuffle, name="input")
                batch_dict = get_batch(BATCH_SIZE, fqueue)
                pointclouds_pl['pointcloud'], labels_pl = batch_dict['frame'], batch_dict['flow']

                smpl = SMPL(SMPL_MODEL_PATH)

                #################for generate 16535 points mesh ######
                edge_for_16535 = np.loadtxt('edge_16535.txt', dtype=np.int)
                edge_for_16535_tf = tf.constant(edge_for_16535, dtype=tf.int32, shape=[9645, 2])

                ##### edge loss and laplacian loss############
                conname = 'laplacian_connect.txt'
                con = np.loadtxt(conname, dtype=np.int)
                edgename = 'edge_for_loss.txt'
                edge = np.loadtxt(edgename, dtype=np.int)

                ##### 108 joint point ######
                p_idx = np.loadtxt('./108idx.txt',dtype=np.int32)
                IDX_JOINT = tf.constant(p_idx, dtype=tf.int32)

                is_training_pl = tf.placeholder(dtype=tf.bool)

                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)

                #####load SMPL ######
                pose = labels_pl[:, :NUM_THETA]
                shape = labels_pl[:, NUM_THETA : NUM_THETA + NUM_SHAPE]

                trans = tf.tile(tf.reshape(labels_pl[:, NUM_THETA+NUM_SHAPE:NUM_THETA+NUM_SHAPE+3], shape=[BATCH_SIZE, 1, 3]), [1, 16535, 1])

                pred_smpl, _ = smpl(shape, pose, trans=trans, v_personal=None)

                label = pointclouds_pl['pointcloud'][:, :NUM_POINT, :3]

                joint_point = tf.gather(pred_smpl, indices=IDX_JOINT, axis=1)
                pointclouds_pl['joint_point'] = joint_point

                print("--- Get model and loss")
                # Get model and loss
                pred, all_data = MODEL.get_model(pointclouds_pl, 
                                    is_training_pl, 
                                    bn_decay=bn_decay)
                
                pred_verts, _ = smpl(shape, pose, trans=trans, v_personal=pred)

                ############## cal loss #########################################
                pred_verts_smpl_16535 = mesh_x(label, edge_for_16535_tf)


                loss =  MODEL.get_loss(pred_verts, pred_verts_smpl_16535)
                # loss = MODEL.get_loss_display(pred_verts, pred_verts_smpl_16535)
                loss = 10*loss #+ 10*loss_template
                
                loss_laplacian = MODEL.Laplacian_loss(label, pred_verts[:, :NUM_POINT, :], con)
                loss_edge = MODEL.edge_loss(label, pred_verts[:, :NUM_POINT, :], edge)
                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)

                train_op = optimizer.minimize(loss + 0.1*(loss_edge + loss_laplacian), global_step=batch)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver(var_list=tf.global_variables())
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
                'pred_smpl': pred_verts_smpl_16535,
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
                print('loss_average', loss_average, loss_average/16535)

                # Save the variables to disk.
                if epoch % 1 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)
            
                    with open(os.path.join(LOG_DIR, "loss.txt"), 'a+') as f:
                        if epoch == 0:
                            f.write('=======================================================\n')
                        f.write('epoch:%d, loss: %.06f \n' % (true_epoch, loss_average/16535))

            coord.request_stop()
            coord.join(threads)
   
def train_one_epoch(sess, ops, saver):
    """ ops: dict mapping from string to tf ops """

    is_training = IS_TRAIN

    num_batches = TRAIN_DATASET // BATCH_SIZE
    loss_sum = 0
    loss_batch = 0

    for batch_idx in range(0, num_batches):


        feed_dict = {
                     ops['is_training_pl']: is_training,
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

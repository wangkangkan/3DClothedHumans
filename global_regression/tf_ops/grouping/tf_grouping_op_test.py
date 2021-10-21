import tensorflow as tf
import numpy as np
from tf_grouping import query_ball_point, group_point

class GroupPointTest(tf.test.TestCase):
  def test(self):
    pass


  def test_grad(self):
    with tf.device('/gpu:0'):
      points = tf.constant(np.reshape(np.loadtxt('point.txt').astype('float32'),[1,2297,3]))
      print(points)
      xyz1 = tf.constant(np.reshape(np.loadtxt('point.txt').astype('float32'),[1,2297,3]))
      xyz2 = tf.constant(np.reshape(np.loadtxt('jp.txt').astype('float32'),[1,24,3]))
      print(xyz1)
      print(xyz2)
      radius = 0.3 
      nsample = 32
      idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
      grouped_points = group_point(points, idx)
      print('111')
      print(grouped_points)

    with self.test_session() as sess:
      print("---- Going to compute gradient error")
      print(idx)
      # print(sess.run(idx))
      print(sess.run(pts_cnt))
      # zero = tf.constant(np.zeros(1,24))
      label = tf.greater(pts_cnt, 0)
      print(sess.run(label))
      print(label)
      label = tf.expand_dims(label,-1)
      label = tf.tile(label,[1,1,32])
      label = tf.cast(label,dtype = tf.int32)
      print(label)
      runlabel = sess.run(label)

      print(runlabel[0,23,:])
      temp = tf.multiply(idx,label)
      print(temp)
      a = sess.run(temp)
      print(a[0,23,:])
      
      # err = tf.test.compute_gradient_error(points, (1,2297,3), grouped_points, (1,24,32,3))
      # print(err)
      # self.assertLess(err, 1e-4) 

  # def test_grad(self):
  #   with tf.device('/gpu:0'):
  #     points = tf.constant(np.random.random((1,128,16)).astype('float32'))
  #     print(points)
  #     xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
  #     xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
  #     radius = 0.3 
  #     nsample = 32
  #     idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
  #     grouped_points = group_point(points, idx)
  #     print(grouped_points)

  #   with self.test_session():
  #     print("---- Going to compute gradient error")
  #     err = tf.test.compute_gradient_error(points, (1,128,16), grouped_points, (1,8,32,16))
  #     print(err)
  #     self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 

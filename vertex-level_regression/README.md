# Stage3: Detail Regression

### Prepare 
Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_m_lbs_10_207_0_v1.0.0.pkl` into `tf_smpl/models`. 

[Download Link](https://pan.baidu.com/s/1N-TsikFeuAqQ8esUqZ2_Xw), extract code：hjun.

### Generate training data
`gene_tfrecord.py`

For generate training data, we need prepare:
  - Detailed model, which is `frame1/x y z` in the function `convert_to_example` in the code.
  - Partial pointcloud from depthmap, which is `frame2/x y z` in the function `convert_to_example` in the code.
  - Predicted SMPL model parameters from stage2, which is `flow/x` in the function `convert_to_example` in the code.

1. Change the data paths in Line 86-89
2. Set the path to save .tfrecord files by changing parameter in Line 78
3. ```
   python gene_tfrecord.py
   ```
### Train and Test

`train.py`

`is_training` for switching between training and testing.


1. Change `--data` in `config_our_3dpoint.py` to your .tfrecord files path.
2. Change `MODEL_PATH` for saving model.
3. ```
   # start training 
   python train.py
   ```


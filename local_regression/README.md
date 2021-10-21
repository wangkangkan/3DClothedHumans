# Stage2 Local Regression
The code framework refers to [HMR](https://github.com/akanazawa/hmr)

### Prepare 
Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_m_lbs_10_207_0_v1.0.0.pkl` into `models`. 

[Download Link](https://pan.baidu.com/s/1N-TsikFeuAqQ8esUqZ2_Xw), extract code：hjun.

### Generate training data
`ourdata_to_tfrecords.py`

For generate training data, we need prepare:
  - Depthmap and camera intrinsic
  - SMPL model parameters
  - Predicted SMPL model parameters from stage1
  - SMPL model
  
1. Change `fx_d` `fy_d` `cx_d` `cy_d` to your camera intrinsic.
2. Get two .txt files which contain the data path and set the parameters in Line123-125
3. Set the path to save .tfrecord files by changing parameter `output_directory`
4. ```
   python ourdata_to_tfrecords.py
   ```
### Train 
`main.py`

1. Get data for the discriminator from [HMR](https://github.com/akanazawa/hmr).
2. Change `DATA_DIR` in `config_our_3dpoint.py` to your .tfrecord files path.
3. ```
   # start training 
   python main.py
   ```

### Test
`demo.py`

[Pre-train model](https://pan.baidu.com/s/1nTX169VqHSkBMVvGwMN30A )

extract code：ghvx

1. Change `fx_d` `fy_d` `cx_d` `cy_d` in `demo.py` to your camera intrinsic.
2. Change `PRETRAINED_MODEL` in `config_our_3dpoint.py` to pre-train model path.
3. ```
   # start testing 
   python demo.py
   ```
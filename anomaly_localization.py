import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random as rn
rn.seed(3407)

import numpy as np
np.random.seed(3407)

import tensorflow as tf
tf.random.set_seed(3407)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras import backend as K
from data_generator import DataGenerator, DataGetter
from H_CAMformer import CAMformer
import os
import sys
import yaml
import numpy as np
import getopt
import tensorflow as tf
import random as rn
import json
from argparse import ArgumentParser
import copy

from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow import keras

def run(config_path):
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    data_train = DataGetter('train', configs['model_opts']).get_data()
    data_test = DataGetter('test', configs['model_opts']).get_data()
    data_val = DataGetter('val', configs['model_opts']).get_data()

    camformer = CAMformer(configs['model_opts'])
    cam_model = camformer.camformer(return_cam=True)
    model_name = configs['model_opts']['model_path']+'/'+configs['model_opts']['model_name']
    cam_model.load_weights(model_name, by_name=False, skip_mismatch=False)
    print("Testing "+model_name+"...")
    results = cam_model.predict(data_test['data'][0])

    seg_weights = results[1]
    cls_weights = results[2]

    preds = results[0]
    preds_levels = [[] for i in range(6)]
    for i in range(len(preds)):
        pred_level = preds[i]
        if i == 0:
            preds_levels[0] = [pred_level for j in range(32)]
        elif i==1 or i==2:
            preds_levels[1] += [pred_level for j in range(16)]
        elif i>2 and i<7:
            preds_levels[2] += [pred_level for j in range(8)]
        elif i>6 and i<15:
            preds_levels[3] += [pred_level for j in range(4)]
        elif i>14 and i<31:
            preds_levels[4] += [pred_level for j in range(2)]
        else:
            preds_levels[5] += [pred_level]


    level_weights = np.asarray([[[0.1667]], [[0.1667]], [[0.1667]], [[0.1667]], [[0.1667]], [[0.1667]]])
    level_weights_seg = np.asarray([[[0.1667]], [[0.1667]], [[0.1667]], [[0.1667]], [[0.1667]], [[0.1667]]])
    level_weights_cls = np.asarray([[[1]], [[0]], [[0]], [[0]], [[0]], [[0]]])
    level_weights_2 = np.asarray([[[0]], [[0]], [[0]], [[0]], [[0]], [[1]]])

    wieghted_average_preds = np.sum(np.asarray(preds_levels)[:,:,:,0]*level_weights, axis=0)

    averaged_seg_weights = np.sum(np.asarray(seg_weights)*level_weights, axis=0)
    averaged_cls_weights = np.sum(np.asarray(cls_weights)*level_weights, axis=0)

    averaged_seg_weights = np.asarray([(averaged_seg_weights[:,i]+averaged_seg_weights[:,i+1])/2.0 for i in range(0,64,2)])
    averaged_cls_weights = np.asarray([(averaged_cls_weights[:,i]+averaged_cls_weights[:,i+1])/2.0 for i in range(0,64,2)])

    averaged_seg_weights = (averaged_seg_weights-np.min(averaged_seg_weights, axis=0, keepdims=True))/\
                           (np.max(averaged_seg_weights, axis=0, keepdims=True)-np.min(averaged_seg_weights, axis=0, keepdims=True))
    averaged_cls_weights = (averaged_cls_weights-np.min(averaged_cls_weights, axis=0, keepdims=True))/\
                           (np.max(averaged_cls_weights, axis=0, keepdims=True)-np.min(averaged_cls_weights, axis=0, keepdims=True))


    averaged_seg_weights = np.stack([np.average(averaged_seg_weights[max(i-1, 0):min(i+2,32),:], axis=0) for i in range(32)])
    averaged_cls_weights = np.stack([np.average(averaged_cls_weights[max(i-1, 0):min(i+2,32),:], axis=0) for i in range(32)])

    averaged_seg_weights = (averaged_seg_weights-np.min(averaged_seg_weights, axis=0, keepdims=True))/\
                           (np.max(averaged_seg_weights, axis=0, keepdims=True)-np.min(averaged_seg_weights, axis=0, keepdims=True))
    averaged_cls_weights = (averaged_cls_weights-np.min(averaged_cls_weights, axis=0, keepdims=True))/\
                           (np.max(averaged_cls_weights, axis=0, keepdims=True)-np.min(averaged_cls_weights, axis=0, keepdims=True))

    max_auc = 0
    acc = 0
    f1 = 0
    prec = 0
    recall = 0
    auc = 0
    alpha = 0.9
    beta =  0.05
    gamma = 0.05
    weights =  alpha*wieghted_average_preds+beta*averaged_seg_weights+gamma*averaged_cls_weights
    weights = np.stack([np.average(weights[max(i-1, 0):min(i+2,32),:], axis=0) for i in range(32)])
    weights_rd = np.round(weights)

    seg_y_pred = (np.transpose(np.where(weights>=0.65, 1, 0)).reshape((weights.shape[0]*weights.shape[1],)).astype(int))
    seg_pred = (np.transpose(weights).reshape((weights.shape[0]*weights.shape[1],)))


    acc = accuracy_score(data_test['data'][2], seg_y_pred)
    f1 = f1_score(data_test['data'][2], seg_y_pred)
    auc = roc_auc_score(data_test['data'][2], seg_pred)
    precision = precision_score(data_test['data'][2], seg_y_pred)
    recall = recall_score(data_test['data'][2], seg_y_pred)
    max_auc = auc

    print("per seqgment performance:")
    print("Accuracy:", acc)
    print("F1-Score:", f1)
    print("AUC:", auc)
    print("Precision:", precision)
    print("Recall:", recall)

if __name__ == '__main__':
    parser = ArgumentParser(description="Anomaly Localization")
    parser.add_argument('--config_file', type=str, help="Path to the directory to load the config file")

    args = parser.parse_args()
    run(args.config_file)

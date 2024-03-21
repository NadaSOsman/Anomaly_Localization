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

from tensorflow.keras.metrics import AUC, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow import keras
from sklearn import metrics


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def run(config_path, test, resume, fusion):
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    data_train = DataGetter('train', configs['model_opts']).get_data()
    data_test = DataGetter('test', configs['model_opts']).get_data()
    data_val = DataGetter('val', configs['model_opts']).get_data()

    if not fusion:
        camformer = CAMformer(configs['model_opts'])
        model = camformer.camformer() if not test else camformer.camformer(return_cam=True)
        model_name = configs['model_opts']['model_path']+'/'+configs['model_opts']['model_name']

        if test or resume:
            print("Lodaing "+model_name+" ...")
            model.load_weights(model_name, by_name=False, skip_mismatch=False)

        if not test:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=configs['model_opts']['lr'],
                                                                      decay_steps=configs['model_opts']['epochs'], decay_rate=0.5)

            loss_dict = {"tf.math.truediv":'binary_crossentropy'}
            for i in range(1,63):
                loss_dict["tf.math.truediv_"+str(i)] = recall_abse_ce

            loss_weights = {"tf.math.truediv":1.0}
            for i in range(1,62):
                loss_weights["tf.math.truediv_"+str(i)] = 1.0

            loss_weights["tf.math.truediv_62"] = 1.0

            metric_dict = {"tf.math.truediv":'accuracy'}
            for i in range(1,63):
                metric_dict["tf.math.truediv_"+str(i)] = f1_m

            optimizer = get_optimizer(configs['model_opts']['optimizer'])(learning_rate=configs['model_opts']['lr'])
            model.compile(loss=loss_dict,
                          loss_weights=loss_weights,
                          optimizer=optimizer,
                          metrics=metric_dict)

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_name,
                                                                     save_weights_only=True,
                                                                     monitor='val_tf.math.truediv_accuracy',
                                                                     mode='max',
                                                                     save_best_only=True)

            history = model.fit(x=data_train['data'][0],
                                y=None,
                                batch_size=configs['model_opts']['batch_size'],
                                epochs=configs['model_opts']['epochs'],
                                validation_data=data_val['data'][0],
                                verbose=1,
                                callbacks=[checkpoint_callback])

            model = camformer.camformer(return_cam=True)
            model.load_weights(model_name, by_name=False, skip_mismatch=False)

    print("Testing ...")
    results = model.predict(data_test['data'][0])[0]
    pred = results[0]
    cam = results[1]

    y_pred = np.where(pred>=0.5, 1.0, 0.0)
    acc = accuracy_score(data_test['data'][1], y_pred)
    f1 = f1_score(data_test['data'][1], y_pred)
    auc_m = AUC()
    auc_m.update_state(data_test['data'][1], pred)
    auc = roc_auc_score(data_test['data'][1], pred)
    precision = precision_score(data_test['data'][1], y_pred)
    recall = recall_score(data_test['data'][1], y_pred)

    print("whole video performance:")
    print("Accuracy:", acc)
    print("F1-Score:", f1)
    print("AUC:", auc)
    print("Precision:", precision)
    print("Recall:", recall)

def get_optimizer(optimizer):
    assert optimizer.lower() in ['adam', 'sgd', 'rmsprop', 'nadam'], \
    "{} optimizer is not implemented".format(optimizer)
    if optimizer.lower() == 'adam':
        return Adam
    elif optimizer.lower() == 'sgd':
        return SGD
    elif optimizer.lower() == 'rmsprop':
        return RMSprop
    elif optimizer.lower() == 'nadam':
        return Nadam



if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for Anomaly Detection")
    parser.add_argument('--config_file', type=str, help="Path to the directory to load the config file")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    run(args.config_file, args.test, args.resume, False)

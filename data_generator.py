import os
import sys
import yaml
import numpy as np
import random as rn
import copy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import vgg16, resnet50, mobilenet
from tensorflow.keras.preprocessing.image import load_img
from pprint import pprint

class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 data_type='train',
                 labels=None,
                 seg_labels=None,
                 data_size=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 opts=None):

        self.data = data
        self.data_type = data_type
        self.labels = labels
        self.seg_labels = seg_labels
        self.batch_size = 1 if len(self.labels) < batch_size  else batch_size
        self.data_size = data_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()
        self.opts = opts
        print(data_type, len(self.data), len(self.labels))

    def get_size(self):
        return len(self.data)

    def __len__(self):
        return int(np.floor(len(self.data)/self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        self.pos_indices = []
        self.neg_indices = []
        for i in range(len(self.data[0])):
            if self.labels[i] == 0.0:
                self.neg_indices.append(i)
            else:
                self.pos_indices.append(i)
        self.neg_indices = np.array(self.neg_indices)
        self.pos_indices = np.array(self.pos_indices)
        if self.shuffle and self.data_type == 'train':
            np.random.shuffle(self.indices)
            np.random.shuffle(self.pos_indices)
            np.random.shuffle(self.neg_indices)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]
        X = self._generate_X(indices)
        Y = self._generate_Y(indices)
        return X, Y

    def _get_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                features = np.load(fid)
            except:
                features = np.load(fid, encoding='bytes')
        return features

    def _generate_X(self, indices):
        X = []
        features_batch = np.empty((self.batch_size, *self.data_size))
        for i, index in enumerate(indices):
            index = int(index)
            if isinstance(self.data[index], str):
                cached_path = self.data[index]
                fiteched_features = self._get_features(cached_path)
                features_batch[i, ] = fiteched_features
            else:
                features_batch[i, ] = self.data[index]
        X.append(features_batch)

        #fetch 32
        features_batch = np.empty((self.batch_size, 32, self.data_size[-1]))
        for i, index in enumerate(indices):
            index = int(index)
            if isinstance(self.data[index], str):
                cached_path = self.data[index]
                fiteched_features = self._get_features(cached_path.replace('slowfast_64', 'slowfast'))
                features_batch[i, ] = fiteched_features
            else:
                features_batch[i, ] = self.data[index]
        X.append(features_batch)

        return X

    def _generate_Y(self, indices):
        Y = np.empty((self.batch_size,))
        for i, index in enumerate(indices):
            Y[i, ] = self.labels[int(index)]

        return [np.round(Y).copy()for i in range(63)]


class DataGetter(object):

    def __init__(self, data_type, model_opts):
        self.model_opts = model_opts
        self._generator = False
        self.data_type = data_type

    def get_data(self):
        self._generator = self.model_opts.get('generator', False)
        data, neg_count, pos_count, labels, seg_labels = self.get_data_names()
        data_size = (self.model_opts['seq_len'], self.model_opts['feat_size'])

        if self.data_type=='val':
            batch_size = self.model_opts['val_batch_size']
        elif self.data_type=='train':
            batch_size = self.model_opts['batch_size']

        if self._generator:
            _data = (DataGenerator(data=data,
                                   data_type=self.data_type,
                                   labels=labels,
                                   seg_labels = np.asarray(seg_labels).reshape(287, 32) if self.data_type=='val' else None,
                                   data_size=data_size,
                                   batch_size=batch_size if (self.data_type=='train' or self.data_type=='val') else 1,
                                   shuffle=self.data_type != 'test',
                                   to_fit=self.data_type != 'test',
                                   opts=self.model_opts), labels, seg_labels)
        else:
            _data = (data, labels, seg_labels)

        return {'data': _data,
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_data_names(self):
        data_path = self.model_opts['data_path']

        data = []
        labels = []
        seg_labels = []
        for file in sorted(os.listdir(os.fsdecode(data_path+"/"+self.data_type+"/"))):
            if 'seg' in os.fsdecode(file):
                seg_labels = np.load(os.fsdecode(data_path+"/"+self.data_type+"/")+os.fsdecode(file))
            elif 'label' in os.fsdecode(file):
                labels = np.load(os.fsdecode(data_path+"/"+self.data_type+"/")+os.fsdecode(file))
            else:
                data.append(os.fsdecode(data_path+"/"+self.data_type+"/")+os.fsdecode(file))

        data = np.asarray(data)

        pos_count = np.sum(labels)
        neg_count = len(labels)-pos_count
        print(self.data_type, ": neg_count =", neg_count, ", pos_count =", pos_count)

        return data, neg_count, pos_count, labels, seg_labels

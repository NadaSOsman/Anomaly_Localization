import os
import sys
import yaml
import numpy as np
import tensorflow as tf
import random as rn
import copy
import math
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate, BatchNormalization, Softmax, Flatten, Add, Activation, Multiply, RepeatVector, AveragePooling1D, MaxPooling1D, ReLU, GaussianNoise
from tensorflow.keras import layers, activations
from tensorflow import keras
from tensorflow.keras.constraints import min_max_norm

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, normalization=True, return_attention_scores=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.normalization = normalization
        self. return_attention_scores = return_attention_scores
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim, activation='relu'),]
        )
        self.layernorm1 = None
        self.layernorm2 = None
        if normalization:
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.normalization = normalization
        self.return_attention_scores= return_attention_scores

    def call(self, inputs, training, attention_mask=None):
        attn_output, att_w = self.att(inputs, inputs, attention_mask=attention_mask, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        if self.normalization:
            out1 = self.layernorm1(inputs + attn_output)
        else:
            out1 = inputs + attn_output
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        if self.normalization:
            out = self.layernorm2(out1 + ffn_output)
        else:
            out = out1 + ffn_output
        #if self.return_attention_scores:
        #    return out, weights
        #else:
        return out, att_w


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = tf.nn.relu(self.pos_emb(positions))
        return x + positions


class CAMformer(object):
    def __init__(self, model_opts=None):
        self.feat_size = model_opts['feat_size']
        self.ff_dim = model_opts['ff_dim']
        self.num_heads = model_opts['num_heads']
        self.MLP_dim = model_opts['MLP_dim']
        self.dropout = model_opts['dropout']
        self.normalize = model_opts['normalize']
        self.seq_len = model_opts['seq_len']
        self.mapping_feat = model_opts['mapping_feat']

    def camformer(self, prev_models=None, return_cam=False):
        input = [Input((self.seq_len, self.feat_size)), Input((int(self.seq_len/2), self.feat_size))]
        in1 = input[0]
        in2 = tf.repeat(input[1], repeats=[2 for i in range(int(self.seq_len/2))], axis=1)
        con_input = in1+in2
        norm_input = layers.LayerNormalization(epsilon=1e-6)(con_input)
        mapped_input = Dense(self.mapping_feat, activation="relu")(norm_input)
        cls_token = tf.random.uniform(shape=[1, 1, self.mapping_feat])*tf.ones_like(Lambda(lambda s:s[:,0:1])(mapped_input))

        # Level 1
        transformer64 = TransformerBlock(self.mapping_feat, self.num_heads, self.ff_dim, normalization=self.normalize)
        out64, wights64 = transformer64(tf.concat([cls_token, mapped_input], axis=1))
        cls_head64 = keras.Sequential([Dense(128, activation='relu'), Dropout(0.1),\
                                       Dense(64, activation='relu'), Dropout(0.1),\
                                       Dense(32, activation='relu'), Dropout(0.1)])

        out_seg64 = tf.nn.softmax(tf.reduce_mean(Lambda(lambda s:s[:,1:])(out64), axis=-1))
        out_cls64 = cls_head64(Lambda(lambda s:s[:,0])(out64))
        cls_weights64 = Lambda(lambda s:s[:,0,1:])(tf.reduce_mean(wights64, axis=1))
        pred64 = (Dense(1, activation='sigmoid')(out_cls64)+Dense(1, activation='sigmoid', use_bias=False)(out_seg64))/2

        # Level 2
        transformer32 = [TransformerBlock(self.mapping_feat, self.num_heads, self.ff_dim, normalization=self.normalize) for i in range(2)]
        out32 = [transformer32[i](tf.concat([cls_token, Lambda(lambda s, i=i:s[:,i*32:i*32+32])(mapped_input)], axis=1)) for i in range(2)]
        cls_head32 = [keras.Sequential([Dense(128, activation='relu'), Dropout(0.1),\
                                        Dense(64, activation='relu'), Dropout(0.1),\
                                        Dense(32, activation='relu'), Dropout(0.1)]) for i in range(2)]

        out_seg32 = [tf.nn.softmax(tf.reduce_mean(Lambda(lambda s:s[:,1:])(out32[i][0]), axis=-1)) for i in range(2)]
        out_cls32 = [cls_head32[i](Lambda(lambda s:s[:,0])(out32[i][0])) for i in range(2)]
        cls_weights32 = [Lambda(lambda s:s[:,0,1:])(tf.reduce_mean(out32[i][1], axis=1)) for i in range(2)]
        pred32_l = [(Dense(1, activation='sigmoid')(out_cls32[i])+Dense(1, activation='sigmoid', use_bias=False)(out_seg32[i]))/2 for i in range(2)]
        pred32 = Dense(1, activation='sigmoid')(tf.concat(pred32_l, axis=-1))

        # Level 3
        transformer16 = [TransformerBlock(self.mapping_feat, self.num_heads, self.ff_dim, normalization=self.normalize) for i in range(4)]
        out16 = [transformer16[i](tf.concat([cls_token, Lambda(lambda s, i=i:s[:,i*16:i*16+16])(mapped_input)], axis=1)) for i in range(4)]
        cls_head16 = [keras.Sequential([Dense(128, activation='relu'), Dropout(0.1),\
                                       Dense(64, activation='relu'), Dropout(0.1),\
                                       Dense(32, activation='relu'), Dropout(0.1)]) for i in range(4)]

        out_seg16 = [tf.nn.softmax(tf.reduce_mean(Lambda(lambda s:s[:,1:])(out16[i][0]), axis=-1)) for i in range(4)]
        out_cls16 = [cls_head16[i](Lambda(lambda s:s[:,0])(out16[i][0])) for i in range(4)]
        cls_weights16 = [Lambda(lambda s:s[:,0,1:])(tf.reduce_mean(out16[i][1], axis=1)) for i in range(4)]
        pred16_l = [(Dense(1, activation='sigmoid')(out_cls16[i])+Dense(1, activation='sigmoid', use_bias=False)(out_seg16[i]))/2 for i in range(4)]
        pred16 = Dense(1, activation='sigmoid')(tf.concat(pred16_l, axis=-1))

        # Level 4
        transformer8 = [TransformerBlock(self.mapping_feat, self.num_heads, self.ff_dim, normalization=self.normalize) for i in range(8)]
        out8 = [transformer8[i](tf.concat([cls_token, Lambda(lambda s, i=i:s[:,i*8:i*8+8])(mapped_input)], axis=1)) for i in range(8)]
        cls_head8 = [keras.Sequential([Dense(128, activation='relu'), Dropout(0.1),\
                                       Dense(64, activation='relu'), Dropout(0.1),\
                                       Dense(32, activation='relu'), Dropout(0.1)]) for i in range(8)]

        out_seg8 = [tf.nn.softmax(tf.reduce_mean(Lambda(lambda s:s[:,1:])(out8[i][0]), axis=-1)) for i in range(8)]
        out_cls8 = [cls_head8[i](Lambda(lambda s:s[:,0])(out8[i][0])) for i in range(8)]
        cls_weights8 = [Lambda(lambda s:s[:,0,1:])(tf.reduce_mean(out8[i][1], axis=1)) for i in range(8)]
        pred8_l = [(Dense(1, activation='sigmoid')(out_cls8[i])+Dense(1, activation='sigmoid', use_bias=False)(out_seg8[i]))/2 for i in range(8)]
        pred8 = Dense(1, activation='sigmoid')(tf.concat(pred8_l, axis=-1))

        # Level 5
        transformer4 = [TransformerBlock(self.mapping_feat, self.num_heads, self.ff_dim, normalization=self.normalize) for i in range(16)]
        out4 = [transformer4[i](tf.concat([cls_token, Lambda(lambda s, i=i:s[:,i*4:i*4+4])(mapped_input)], axis=1)) for i in range(16)]
        cls_head4 = [keras.Sequential([Dense(128, activation='relu'), Dropout(0.1),\
                                       Dense(64, activation='relu'), Dropout(0.1),\
                                       Dense(32, activation='relu'), Dropout(0.1)]) for i in range(16)]

        out_seg4 = [tf.nn.softmax(tf.reduce_mean(Lambda(lambda s:s[:,1:])(out4[i][0]), axis=-1)) for i in range(16)]
        out_cls4 = [cls_head4[i](Lambda(lambda s:s[:,0])(out4[i][0])) for i in range(16)]
        cls_weights4 = [Lambda(lambda s:s[:,0,1:])(tf.reduce_mean(out4[i][1], axis=1)) for i in range(16)]
        pred4_l = [(Dense(1, activation='sigmoid')(out_cls4[i])+Dense(1, activation='sigmoid', use_bias=False)(out_seg4[i]))/2 for i in range(16)]
        pred4 = Dense(1, activation='sigmoid')(tf.concat(pred4_l, axis=-1))

        # Level 6
        transformer2 = [TransformerBlock(self.mapping_feat, self.num_heads, self.ff_dim, normalization=self.normalize) for i in range(32)]
        out2 = [transformer2[i](tf.concat([cls_token, Lambda(lambda s, i=i:s[:,i*2:i*2+2])(mapped_input)], axis=1)) for i in range(32)]
        cls_head2 = [keras.Sequential([Dense(128, activation='relu'), Dropout(0.1),\
                                       Dense(64, activation='relu'), Dropout(0.1),\
                                       Dense(32, activation='relu'), Dropout(0.1)]) for i in range(32)]

        out_seg2 = [tf.nn.softmax(tf.reduce_mean(Lambda(lambda s:s[:,1:])(out2[i][0]), axis=-1)) for i in range(32)]
        out_cls2 = [cls_head2[i](Lambda(lambda s:s[:,0])(out2[i][0])) for i in range(32)]
        cls_weights2 = [Lambda(lambda s:s[:,0,1:])(tf.reduce_mean(out2[i][1], axis=1)) for i in range(32)]
        pred2_l = [(Dense(1, activation='sigmoid')(out_cls2[i])+Dense(1, activation='sigmoid', use_bias=False)(out_seg2[i]))/2 for i in range(32)]
        pred2 = Dense(1, activation='sigmoid')(tf.concat(pred2_l, axis=-1))


        if return_cam:
            preds = [pred64]+pred32_l+pred16_l+pred8_l+pred4_l+pred2_l
            out_segs = [out_seg64, tf.concat(out_seg32, axis=-1), tf.concat(out_seg16, axis=-1), tf.concat(out_seg8, axis=-1),\
                        tf.concat(out_seg4, axis=-1), tf.concat(out_seg2, axis=-1)]
            cls_weights = [cls_weights64, tf.concat(cls_weights32, axis=-1), tf.concat(cls_weights16, axis=-1), tf.concat(cls_weights8, axis=-1),\
                       tf.concat(cls_weights4, axis=-1), tf.concat(cls_weights2, axis=-1)]
            model = Model(input, [preds, out_segs, cls_weights], name='hcamformer')
        else:
            preds = [pred64]+pred32_l+pred16_l+pred8_l+pred4_l+pred2_l
            model = Model(input, preds, name='hcamformer')

        return model

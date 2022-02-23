#!/usr/bin/env python
# coding: utf-8

# todo: 
# 1. + попробовать пулинг по каналам
# 2. попробовать увеличить алфавит добавить е и т
# 3. чекнуть как свитается цер
import bz2
import gc
from sklearn.model_selection import ParameterGrid
import numpy as np
from config import SEQ_LENGTH, FRAMERATE, CHUNK, FFT_SIZE
import generate_wav_samples as gen
import os
from config import MORSE_CHR
from tqdm import tqdm
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Input, Dense, Activation,TimeDistributed, GlobalMaxPooling1D
from tensorflow.keras.layers import Reshape, Lambda, Dropout, Bidirectional, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, SimpleRNN,LSTM
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow.keras.callbacks
import pickle
import Levenshtein

import generator_test as gt

OUTPUT_DIR = 'rnn_output'

class VizCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, run_name, test_func, X):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.X = X

    def show_edit_distance(self, num):
        print('edit distance: ', num)
        """
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func,
                                       word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j],
                                              word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance:'
              '%.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))
        """

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        
        #self.show_edit_distance(256)
        
        dec_len = 10
        for i in range(dec_len):
            labels = self.X[1][i:i+1]
            print('labels: ', labels_to_text([int(e) for e in labels[0]]))
        
        word_batch = self.X[0][:dec_len]
        res = decode_batch(self.test_func, word_batch)
        print('result lens: ', len(res))
        for e in res[:dec_len]:
            print(e)
        
        len_for_cer_count = 5000
        word_batch = self.X[0][:len_for_cer_count]
        res = decode_batch(self.test_func, word_batch)
        print()
        
        cers = []
        for i, t in enumerate(self.X[1][:len_for_cer_count]):
            true = labels_to_text(t)
            pred = res[i]

            c = cer(true, pred)

            cers.append(c)

        print(np.mean(cers))
            
def cer(true, pred):
    t = ''.join(true).strip()
    p = ''.join(pred).strip()
    distance = Levenshtein.distance(t, p)
    return distance / len(t) if len(t) > 0 else len(p)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    bc = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    return bc


def labels_to_text(i):
    return [MORSE_CHR[e] for e in i]

def decode_batch2(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    print(np.argmax(out, axis = -1))
    return np.argmax(out, axis = -1)


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    r = np.argmax(out, axis=-1)
    #print('r: ', r)
    
    res = []
    for a in r:
        sub_res = []
        for i, e in enumerate(a):
            #print(i, e)
            if i == 0:
                sub_res.append(e)
                continue
            if (e == a[i-1]):
                continue
            if (e == len(MORSE_CHR) - 1):
                continue
            sub_res.append(e)
            
        sub_res = [e for e in sub_res if e != len(MORSE_CHR) - 1]
        sub_res = labels_to_text(sub_res)
        res.append(sub_res)
            
    #[e if (i==0 or c != bc[i-1] and c!=3)]
    #print('res: ', res)
    return res

def get_wide_model(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    prev = None
    for i in range(arch['conv_blocks']):
        conv = Conv1D(arch['conv_filters'], arch['kernel_size'], strides=1, padding='same',
                      activation=act, kernel_initializer='he_normal',
                      name=f'conv_{i}')(input_data if prev is None else prev)

        # pm1 = Permute((2, 1))(inner)
        # mp = MaxPooling1D(pool_size=pool_size, name=f'max_{i}', strides=2, padding='same', )(pm1)
        # pm2 = Permute((2, 1))(mp)
        # mp = MaxPooling1D(pool_size=pool_size, name=f'max_{i}', strides=1, padding='same', )(inner)
        # ll = Lambda(channelPool)(inner)#, output_shape=optionalInTensorflow
        prev = conv

    conv10 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10')(input_data)

    conv11 = Conv1D(6, 16, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_11')(conv10)

    conv12 = Conv1D(8, 32, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_12')(conv11)

    conv13 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_13')(conv12)


    # gru2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(mp)
    # gru = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru1',unroll=True)(mp)
    # dense0 = Dense(16, kernel_initializer='he_normal', name='dense0')(mp)

    srnn = SimpleRNN(64, return_sequences=True, kernel_initializer='he_normal')(conv13)
    # srnn2 = SimpleRNN(32, return_sequences=True, kernel_initializer='he_normal',)(srnn)
    # srnn3 = SimpleRNN(32, return_sequences=True, kernel_initializer='he_normal',)(srnn2)

    # lstm = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm',unroll=True)(mp)
    # lstm2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm)

    # lstm2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm)
    # lstm3 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm3')(lstm2)

    # dense2 = Dense(128, kernel_initializer='he_normal', name='dense2')(gru)

    # dpo = Dropout(0.01, name='do1')(gru)
    dense1 = Dense(dict_len, kernel_initializer='he_normal', name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    print(y_pred, labels, input_length, label_length)

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v1(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv10 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_10')(input_data)

    conv11 = Conv1D(6, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_11')(conv10)

    conv12 = Conv1D(8, 32, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(conv11)

    conv13 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_13')(conv12)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv13)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v5(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv10 = Conv1D(4, 4, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_10')(input_data)

    conv11 = Conv1D(4, 4, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_11')(conv10)

    conv12 = Conv1D(8, 32, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(conv11)

    conv13 = Conv1D(8, 32, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_13')(conv12)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv13)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v6(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv12 = Conv1D(8, 32, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(input_data)

    conv13 = Conv1D(8, 32, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_13')(conv12)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv13)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v7(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv12 = Conv1D(16, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(input_data)

    conv13 = Conv1D(16, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_13')(conv12)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv13)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v8(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv12 = Conv1D(32, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(input_data)

    conv13 = Conv1D(32, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_13')(conv12)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv13)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v9(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv12 = Conv1D(32, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(input_data)


    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv12)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v10(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv12 = Conv1D(16, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(input_data)


    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv12)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v11(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv12 = Conv1D(8, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(input_data)


    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv12)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v12(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv12 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(input_data)


    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv12)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v13(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(input_data)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v2(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv10 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_10')(input_data)

    conv11 = Conv1D(6, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_11')(conv10)

    conv12 = Conv1D(16, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(conv11)

    conv13 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_13')(conv12)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv13)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v4(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv10 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_10')(input_data)

    conv11 = Conv1D(6, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_11')(conv10)

    conv12 = Conv1D(16, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(conv11)

    conv13 = Conv1D(16, 32, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_13')(conv12)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv13)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v3(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv10 = Conv1D(4, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_10')(input_data)

    conv11 = Conv1D(6, 16, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_11')(conv10)

    conv12 = Conv1D(16, 64, padding='same',
                    activation=act, kernel_initializer=arch['kernel_initializer'],
                    name=f'conv_12')(conv11)


    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv12)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v14(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 64, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 64, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    conv3 = Conv1D(8, 64, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_3', dilation_rate=1)(conv2)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv3)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v15(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 64, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 64, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v150(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 32, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v150gru(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 32, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = GRU(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb



def get_progression_model_v15001(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(20, 32, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(20, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v150011(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(20, 48, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(20, 48, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v150012(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(20, 64, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(20, 64, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v150013(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = arch['activation'] #'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(20, 64, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(20, 96, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v15002(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(12, 32, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v1501(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(8, 32, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(8, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v1502(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(16, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 32, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v1503(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 16, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(4, 32, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(4, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v151(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 64, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 128, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v152(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(8, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 64, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 128, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v153(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 64, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(32, 128, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb

def get_progression_model_v154(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 64, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 128, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 128, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv2)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v16(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(16, 128, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    conv2 = Conv1D(16, 128, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_2', dilation_rate=1)(conv10)

    conv3 = Conv1D(8, 64, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_3', dilation_rate=1)(conv2)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv3)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v17(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(4, 32, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    conv10 = Conv1D(8, 512, strides=1, padding='same',
                    activation=act, kernel_initializer='he_normal',
                    name=f'conv_10', dilation_rate=1)(conv1)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv10)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def get_progression_model_v18(optimizer, arch):
    input_shape = (mel_len, mel_count)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    conv1 = Conv1D(16, 1024, strides=1, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name=f'conv_1', dilation_rate=1)(input_data)

    srnn = SimpleRNN(arch['rnn_size'], return_sequences=True, kernel_initializer=arch['kernel_initializer'])(conv1)

    dense1 = Dense(dict_len, kernel_initializer=arch['kernel_initializer'], name='dense1')(srnn)

    y_pred = Activation('softmax', name='softmax')(dense1)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    test_func = K.function([input_data], [y_pred])
    viz_cb = VizCallback('test', test_func, X_val)

    return model, viz_cb


def main():
    global dict_len, max_seq_len, mel_count, mel_len, X_val, conv_filters, kernel_size
    # In[4]:
    sample_len = SEQ_LENGTH
    samples_count = 100000
    sr = 8000
    dict_len = len(MORSE_CHR)
    max_seq_len = 5
    mel_count = 1
    mel_len = 161
    # In[5]:
    dg = gen.DataGenerator()
    g = dg.seq_generator(SEQ_LENGTH, FRAMERATE, 1, sr, mel_count)

    # In[6]:
    def read_data(set_len, g):
        l = np.zeros([set_len, max_seq_len], dtype=np.int32)
        X = np.zeros([set_len, mel_len, mel_count])
        input_length = np.zeros([set_len, 1], dtype=np.int32)
        label_length = np.zeros([set_len, 1], dtype=np.int32)

        i = 0
        for wave, label_indexes, labels, c, mel in tqdm(g):
            if len(labels) > max_seq_len:
                continue

            X[i, :, :] = mel

            l[i, :len(labels)] = labels
            input_length[i, :] = mel.shape[0]

            label_length[i, :1] = c

            i += 1
            if i == set_len:
                break

        return [X, l, input_length, label_length], l


    dataset = 'dataset_500k_full_wpm'
    if True:
        with bz2.BZ2File(f'{dataset}.pbz2', 'r') as f:
            X, l = pickle.load(f)

    X_val, l_val = read_data(200, g)

    conv_filters = 64
    kernel_size = 16
    conv_blocks = 2
    pool_size = 32
    time_dense_size = 32
    rnn_size = 64
    minibatch_size = 32
    lr = 0.005

    aaarch = {
        'conv_filters': conv_filters,
        'kernel_size': kernel_size,
        'conv_blocks': conv_blocks,
        'rnn_size': rnn_size,
        'lr': lr,
        'kernel_initializer': 'he_normal'
    }

    progression_grid = {
        'rnn_size': [32],
        'lr': [0.005],
        'kernel_initializer': ['he_normal'], # ['he_normal', 'normal', 'uniform']
        'bs': [128],
        'activation': ['linear', 'elu', 'selu', 'tanh',]
    }

    models = {
        #'progression_v1': (lambda arch: get_progression_model_v1(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v2': (lambda arch: get_progression_model_v2(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v3': (lambda arch: get_progression_model_v3(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v4': (lambda arch: get_progression_model_v4(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v5': (lambda arch: get_progression_model_v5(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v6': (lambda arch: get_progression_model_v6(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v7': (lambda arch: get_progression_model_v7(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v8': (lambda arch: get_progression_model_v8(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v9': (lambda arch: get_progression_model_v9(RMSprop(lr=arch['lr']), arch), progression_grid),

        #'progression_v14': (lambda arch: get_progression_model_v14(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v150': (lambda arch: get_progression_model_v150(RMSprop(lr=arch['lr']), arch), progression_grid),
        'progression_v150011': (lambda arch: get_progression_model_v150011(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v150013': (lambda arch: get_progression_model_v150013(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v15002': (lambda arch: get_progression_model_v15002(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v150gru': (lambda arch: get_progression_model_v150gru(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v1502': (lambda arch: get_progression_model_v1502(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v1503': (lambda arch: get_progression_model_v1503(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v151': (lambda arch: get_progression_model_v151(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v152': (lambda arch: get_progression_model_v152(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v153': (lambda arch: get_progression_model_v153(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v154': (lambda arch: get_progression_model_v154(RMSprop(lr=arch['lr']), arch), progression_grid),

        #'progression_v16': (lambda arch: get_progression_model_v16(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v17': (lambda arch: get_progression_model_v17(RMSprop(lr=arch['lr']), arch), progression_grid),
        #'progression_v18': (lambda arch: get_progression_model_v18(RMSprop(lr=arch['lr']), arch), progression_grid),
    }

    for model_name, (model_lambda, grid) in models.items():
        for arch in ParameterGrid(grid):
            for try_index in range(3):

                model, viz_cb = model_lambda(arch)
                params_count = model.count_params()

                gc.collect()

                fit_history_0 = model.fit([e[:50000] for e in X], l[:50000], validation_split=0.025, batch_size=1000, callbacks=[viz_cb], epochs=5)
                fit_history = model.fit(X, l, validation_split=0.025, batch_size=arch['bs'], callbacks=[viz_cb], epochs=4)
                model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=RMSprop(lr=0.0005))
                fit_history2 = model.fit(X, l, validation_split=0.025, batch_size=64, callbacks=[viz_cb], epochs=1)

                print(fit_history.history)

                best_loss = np.min(fit_history2.history['loss'])
                best_val = np.min(fit_history2.history['val_loss'])

                open('output/nas_results_3.txt', 'a+').write(f'{model_name}\t{try_index}\t{best_val}'
                                                    f'\t{best_loss}\t{arch}\t{fit_history.history}\t{params_count}\t{fit_history2.history}\n')


if __name__ == '__main__':
    main()






#######################################################################################################################
# Update V8_7:
#   - Adding masking weights to the dataset and revising calculate_prediction function. With this change,
#     we can filter scenes and time stamps in both training or testing (it was not possible before on training).
########################################################################################################################
# Shahriar S. Heydari, 12/21/2020

import os, fnmatch
import multiprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: no INFO, 2: no INFO and WARNING, 3: no INFO, WARNING, and ERROR printed
import numpy as np
import math, time, glob, random, sys, re, warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Default parameter values (will be overridden by set_parameters function if input configuration file is provided)
inFolder = '/home/sshahhey/TFRecord/allblocks2/'
outFolder = '/home/sshahhey/WIP/'
baseModelFolder = ''
baseModels = ['']
#files = sorted(os.listdir(baseModelFolder))
#baseModels = fnmatch.filter(files, '*best_model.h5')
resumeFlag = 'No'  # 'No', 'weights', 'full', or 'last'
set_no = [0]
gpu_no = '0'
pid_prefix = 'allblocks_full'
trainFilePattern = ['*']
testFilePattern = ['*']
fixed_recurrent_features = [0, 1, 2, 3, 4, 5, 6, 7, 10, 17, 37, 30]
fixed_annual_features = []
fixed_static_features = [0, 1, 2]
testFeatures = [0]
sensors = [.5,.7,.8]
reduceTime = 0      # if 0, full year sdequences are used,
                    # If 1, the time range between the first and second elements of selectDOY are used,
#                   # if 2, the observation closest to selectDOY elements are used.
selectDOY = [180]  # (15, 106, 197, 288 for 4 seasons)
selectDOYs = np.array(selectDOY)/366
mode = 'test'
max_epochs = 100
min_epochs = 50
es_flag = 25  # early stopping criteria, 0 = disable
train_batch = 1024
train_reduce = 10
val_reduce = 10
test_reduce = 1
RLayers = (48,48,48)
DLayers1 = (16,16)
DLayers2 = (32,32)
Dropouts = (0.25,0.25,0.25,0.1,0.1,0,0)
optimizer_type = 'Adam'
optimizer_args = {'lr': 0.001, 'amsgrad':True}
kernel_reg = ker_reg = None  # Kernel regularization, first variable is just the string identifier
recurrent_reg = rec_reg = None  # Recurrent regularization, first variable is just the string identifier
dense_act = 'tanh'  # Dense layers activation function (for LSTM layers it is fixed by default as tanh)
generateF1 = True
cacheTrainDataset = True
cacheValDataset = True
float16_flag = True
extendedLogTest = 0         # 0: Don't do extended False, 1: Enable it but don't cache TestDataset,
                            # 2: Enable it and do cache TestDataset
verbose = 2  # 1 = training progress indicator, 2 = summary only, 0 = None
log_level = 0  # 0 = All information is logged, 1= Important information is logged

if generateF1:
    F_eval_set = 'test'
else:
    F_eval_set = None

# Reading input configuration file and command line parameters and set the variables
def set_parameters(parameters):
    global inFolder, set_no, outFolder, pid_prefix, train_reduce, val_reduce, test_reduce, \
        RLayers, DLayers1, DLayers2, Dropouts, kernel_reg, rec_reg, optimizer_type, optimizer_args, kernel_reg, \
        recurrent_reg, dense_act, trainFilePattern, testFilePattern, testFeatures, train_batch, gpu_no, \
        fixed_recurrent_features, fixed_annual_features, fixed_static_features, float16_flag, \
        generateF1, min_epochs, max_epochs, es_flag, cacheTrainDataset, cacheValDataset, extendedLogTest

    def rangeexpand(txt):
        lst = []
        if txt:
            for r in txt.split(','):
                if '-' in r[1:]:
                    r0, r1 = r[1:].split('-', 1)
                    lst += range(int(r[0] + r0), int(r1) + 1)
                else:
                    lst.append(int(r))
        return lst

    dic = {}
    config_file = parameters[1]
    with open(config_file, 'r') as f:
        for line in f:
            try:
                (key, val) = line.split('=')
                dic[key.strip()] = val.strip()
            except:
                pass
    inFolder = dic['inFolder']
    outFolder = dic['outFolder']
    set_no = rangeexpand(dic['set_no'])
    gpu_no = dic['gpu_no']
    pid_prefix = dic['pid_prefix']
    trainFilePattern = eval(dic['trainFilePattern'])
    testFilePattern = eval(dic['testFilePattern'])
    fixed_recurrent_features = rangeexpand(dic['fixedRecurrentFeatures'])
    fixed_annual_features = rangeexpand(dic['fixedAnnualFeatures'])
    fixed_static_features = rangeexpand(dic['fixedStaticFeatures'])
    testFeatures = rangeexpand(dic['testFeatures'])
    min_epochs = int(dic['min_epochs'])
    max_epochs = int(dic['max_epochs'])
    es_flag = int(dic['es_flag'])
    train_batch = int(dic['train_batch'])
    train_reduce = float(dic['train_reduce'])
    val_reduce = float(dic['val_reduce'])
    test_reduce = float(dic['test_reduce'])
    RLayers = eval(dic['RLayers'])
    DLayers1 = eval(dic['DLayers1'])
    DLayers2 = eval(dic['DLayers2'])
    Dropouts = eval(dic['Dropouts'])
    optimizer_type = dic['optimizer_type']
    optimizer_args = eval(dic['optimizer_args'])
    kernel_reg = dic['kernel_reg']
    recurrent_reg = dic['recurrent_reg']
    dense_act = dic['dense_act']
    generateF1 = eval(dic['generateF1'])
    cacheTrainDataset = eval(dic['cacheTrainDataset'])
    cacheValDataset = eval(dic['cacheValDataset'])
    float16_flag = eval(dic['float16_flag'])
    extendedLogTest = int(dic['extendedLogTest'])
    if len(parameters) > 2:
        inFolder = parameters[2]
        outFolder = parameters[3]
    if len(parameters) > 4:
        train_reduce = float(parameters[4])
        val_reduce = float(parameters[5])
        test_reduce = float(parameters[6])
    if len(parameters) > 7:
        min_epochs = int(parameters[7])
        max_epochs = int(parameters[8])
        es_flag = int(parameters[9])
    if len(parameters) > 10:
        generateF1 = eval(parameters[10])
        cacheTrainDataset = eval(parameters[11])
        cacheValDataset = eval(parameters[11])
        float16_flag = eval(parameters[12])
        extendedLogTest = int(parameters[13])
    if len(parameters) > 14:
        fixed_recurrent_features = fixed_recurrent_features + rangeexpand(
            parameters[14].replace('[', '').replace(']', ''))

if len(sys.argv) > 1:
    set_parameters(sys.argv)

if not Dropouts:            # Dropout should be set either to all zero (disabled) or something else
    Dropouts = np.zeros(len(RLayers) + len(DLayers1) + len(DLayers2))
else:
    if len(Dropouts) != len(RLayers) + len(DLayers1) + len(DLayers2):
        print('Dropout ratios list does not match existing layers')
        sys.exit(0)

# Setup Tensorflow and its related modules
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: no INFO, 2: no INFO and WARNING, 3: no INFO, WARNING, and ERROR printed
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

config = ConfigProto()
config.gpu_options.allow_growth = True
if kernel_reg:
    ker_reg = eval(kernel_reg)
else:
    ker_reg = None
if recurrent_reg:
    rec_reg = eval(recurrent_reg)
else:
    rec_reg = None

# Other common parameters that are changed rarely
lastBlock = 'samp84'
NC = 7  # Default number of allowed landcover classes (from 21 to 27)
starting_class = 21  # Default land cover code of the first class
maxSeqLen = 100  # Maximum length of input sequences over all blocks
minDelta = 0.005
DELTAS = [minDelta, -minDelta]  # minimum accuracy and loss delta used in training stopping
dual_graph_output = False
test_batchSize = 1024
createProgressFile = True

# Below variables will reassigned based on dataset configuration
trainFiles = []
testFiles = []
neighborhood = 1
num_recurrent_features = num_annual_features = num_static_features = 0  # Number of input features (will be assigned later from config file)
selected_recurrent_features = []
selected_annual_features = []
selected_static_features = []
selected_model = ''
binaryFlag = False
remove_dup = False
inputdata_float16 = False

# Input TFRecord dataset structure definition dictionary
featuresDict = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
                'annualData': tf.io.FixedLenFeature([], dtype=tf.string),
                'staticData': tf.io.FixedLenFeature([], dtype=tf.string),
                'rows': tf.io.FixedLenFeature([], dtype=tf.int64),
                'label': tf.io.FixedLenFeature([], dtype=tf.int64)
                }

def parse_tfrecord(example, dataset_type):
    def closest(lst, K):
        idx = tf.compat.v1.arg_min(tf.abs(lst - K),0)
        return idx
    def filterTime(input):
        recurrentData = input
        if reduceTime == 2:
            index = [closest(input[:, 0], sd) for sd in selectDOYs]
            recurrentData = tf.gather(input, index, axis=0)
        elif reduceTime == 1:
            start = tf.greater_equal(input[:, 0], selectDOYs[0])
            end = tf.greater_equal(tf.negative(input[:, 0]), -selectDOYs[1])
            recurrentData = tf.boolean_mask(input, tf.logical_and(start, end))
        return recurrentData

    # read example
    features = tf.io.parse_single_example(serialized=example, features=featuresDict)
    label = tf.one_hot(features['label'] - starting_class, NC)
    rows = features['rows']
    # extract sequence data part
    recurrentData = tf.io.decode_raw(features['data'], tf.float32)
    if float16_flag:
        recurrentData = tf.cast(recurrentData, tf.float16)
    recurrentData = tf.reshape(recurrentData, (rows, num_recurrent_features))
    # pick the desired features
    recurrentData = tf.gather(recurrentData, selected_recurrent_features, axis=1)
    # filter based on Landsat sensor
    sensor_list = recurrentData[:,1]
    mask = tf.greater(sensor_list,1.0)
    if 0.5 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.5))
    if 0.7 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.7))
    if 0.8 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.8))
    recurrentData = tf.boolean_mask(recurrentData, mask)
    idx = tf.shape(recurrentData)[0]
    # if not empty sequence, filter time span if needed
    recurrentData = tf.cond(tf.equal(idx,0), lambda: recurrentData, lambda: filterTime(recurrentData))
    # pad the sequence to to maximum length to have all examples in a batch having the same size
    dataLen = tf.shape(input=recurrentData)[0]
    weight = tf.cond(tf.equal(dataLen,0), lambda: 0, lambda: 1)
    recurrentData = tf.pad(tensor=recurrentData, paddings=[[maxSeqLen - dataLen, 0], [0, 0]])
    # extract annual features if any annual feature is selected
    if selected_annual_features:
        annualData = tf.gather(tf.io.decode_raw(features['annualData'], tf.float32), selected_annual_features)
        if float16_flag:
            annualData = tf.cast(annualData, tf.float16)
    else:
        annualData = []
    # extract static features if any static feature is selected
    if selected_static_features:
        staticData = tf.gather(tf.io.decode_raw(features['staticData'], tf.float32), selected_static_features)
        if float16_flag:
            staticData = tf.cast(staticData, tf.float16)
    else:
        staticData = []
    # pack the data into dictionary and return it with its class label
    if dataset_type == 'test':
        return {'recurrentData': recurrentData, 'annualData': annualData, 'staticData': staticData,
                'seq_len': dataLen}, label, weight
    else:
        return {'recurrentData': recurrentData, 'annualData': annualData, 'staticData': staticData}, label, weight

def read_datasets(pattern, numFiles, numEpochs=None, type='train', batchSize=None, take=1, cache=False):
    files = tf.data.Dataset.list_files(pattern)

    def _parse(x):
        x = tf.data.TFRecordDataset(x, compression_type='GZIP')
        return x

    dataset = files.interleave(_parse, cycle_length=numFiles, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(lambda x: parse_tfrecord(x, type), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.take(take)                # will take only this number of examples from dataset
    dataset = dataset.batch(batchSize)
    if cache:
        print('cache enabled for take=',take)
        dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(numEpochs)
    return dataset

def keras_model(input_shape1, input_shape2, input_shape3):
    from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
    from tensorflow.compat.v1.keras.layers import CuDNNLSTM

    input_recurrent = Input(shape=input_shape1, name='recurrentData')
    input_annual = Input(shape=input_shape2, name='annualData')
    input_static = Input(shape=input_shape3, name='staticData')
    input_nonrecurrent = concatenate([input_annual, input_static])

    x = input_recurrent
    for i in range(len(RLayers) - 1):
        x = CuDNNLSTM(RLayers[i], return_sequences=True, kernel_regularizer=ker_reg, recurrent_regularizer=rec_reg)(x)
        if Dropouts[i] > 0:
            x = Dropout(Dropouts[i])(x)
    x = CuDNNLSTM(RLayers[-1], kernel_regularizer=ker_reg, recurrent_regularizer=rec_reg)(x)
    if Dropouts[len(RLayers) - 1] > 0:
        x = Dropout(Dropouts[len(RLayers) - 1])(x)

    if input_shape2[0] + input_shape3[0] != 0:
        y = input_nonrecurrent
        for j in range(len(DLayers1)):
            y = Dense(DLayers1[j], activation=dense_act, kernel_regularizer=ker_reg)(y)
            if Dropouts[len(RLayers) + j] > 0:
                y = Dropout(Dropouts[len(RLayers) + j])(y)
        z = concatenate([x, y])
    else:
        z = x

    for j in range(len(DLayers2)):
        z = Dense(DLayers2[j], activation=dense_act, kernel_regularizer=ker_reg)(z)
        if Dropouts[len(RLayers) + len(DLayers1) + j] > 0:
            z = Dropout(Dropouts[len(RLayers) + len(DLayers1) + j])(z)

    outputs = Dense(NC, activation='softmax')(z)

    return tf.keras.Model([input_recurrent, input_annual, input_static], outputs)

# def calculate_AUC(y,scores, NC):
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(NC):
#         fpr[i], tpr[i], _ = roc_curve(y[:, i], scores[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), scores.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#     # Compute macro-average ROC curve and ROC area
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NC)]))
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(NC):
#         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#     # Finally average it and compute AUC
#     mean_tpr /= NC
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#     # Plot all ROC curves
#     plt.figure()
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-avg'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle='--', linewidth=2)
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-avg'
#                    ''.format(roc_auc["macro"]),
#              color='navy', linestyle='--', linewidth=2)
#     #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#     for i in range(NC):#, color in zip(range(NC), colors):
#         plt.plot(fpr[i], tpr[i],# color=color,
#                  label='class {0}'
#                        ''.format(i, roc_auc[i]))
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Multi-class Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     #plt.show()
#     return roc_auc, plt
#
def calculate_predictions(model, data_iterator, eval_steps):

    y_eval = y_pred = seq_len = np.empty(0)
    y_pred_prob = np.empty((0,NC))
    for i in range(eval_steps):
        # read data batch and store its parts in different variables
        batch = next(data_iterator)
        seq_len_batch = batch[0]['seq_len']
        del batch[0]['seq_len']
        x_eval_batch = batch[0]
        y_eval_batch = batch[1]
        weights = batch[2]
        # calculate model predictions
        y_pred_batch = model.predict_on_batch(x_eval_batch)
        # locate samples with zero weight...
        b0 = np.where(weights==0)[0]
        # ... and remove their corresponding prediction and references
        y_pred_batch = np.delete(y_pred_batch,b0,axis=0)
        y_eval_batch = np.delete(y_eval_batch,b0,axis=0)
        seq_len_batch = np.delete(seq_len_batch,b0)
        # append resulting data to the previous batch data
        seq_len = np.append(seq_len, seq_len_batch)
        y_eval = np.append(y_eval, np.argmax(y_eval_batch, axis=1))
        y_pred_prob = np.vstack((y_pred_prob, y_pred_batch))
        y_pred = np.append(y_pred, np.argmax(y_pred_batch, axis=1))
        if i % 100 == 0:
            print(' - Evaluating {} of {} records done'.format(i+1, eval_steps))
    return y_eval, y_pred, y_pred_prob, seq_len

class MakeFile(tf.keras.callbacks.Callback):
    def __init__(self,
                 outFolder='',
                 file_name='',
                 num_param=0,
                 eval_data_iterator=None,
                 eval_steps=0):
        super(MakeFile, self).__init__()

        self.outFolder = outFolder
        self.file_name = file_name
        self.iterate_file = None
        self.prev_file = None
        self.best_acc = 0
        self.best_val_acc = 0
        self.best_val_loss = 0
        self.best_epoch = 0
        self.accuracy_table = []
        self.start_time = 0
        self.log_status = 'WIP'
        self.num_param = num_param / 1000000.0
        self.eval_data_iterator = eval_data_iterator
        self.eval_steps = eval_steps

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        eval_accuracy = 0
        eval_avg_f1 = 0
        f1_min_value = 0
        f1_min_class = '0.0'
        test_time = 0
        improvement = False
        if (logs['val_accuracy'] > self.best_val_acc):
            improvement = True
            self.best_val_acc = logs['val_accuracy']
            self.best_acc = logs['accuracy']
            self.best_epoch = epoch + 1
        if (epoch + 1 == min_epochs) or \
                ((epoch + 1 > min_epochs) and (extendedLogTest > 0) and improvement):
            start_time = time.time()
            y_eval, y_pred, _, _ = calculate_predictions(self.model, self.eval_data_iterator, self.eval_steps)
            report = classification_report(y_eval, y_pred, digits=5, output_dict=True)
            eval_accuracy = report['accuracy']
            eval_avg_f1 = report['macro avg']['f1-score']
            unwanted = {'accuracy', 'macro avg', 'weighted avg'}
            for unwanted_key in unwanted: del report[unwanted_key]
            f1_list = [report[key]['f1-score'] for key in report.keys()]
            f1_min_index = f1_list.index(min(f1_list))
            f1_min_value = f1_list[f1_min_index]
            f1_min_class = list(report.keys())[f1_min_index]
            test_time = time.time() - start_time
        self.accuracy_table.append((logs['accuracy'] ,logs['val_accuracy'], logs['val_loss'], time.time()-self.start_time,
                                    eval_accuracy, eval_avg_f1, f1_min_value, f1_min_class, test_time))
        file_path = outFolder + "AA_{:.3f}_{:.3f}_{}_{}_{}_{:.2f}M_".format(100*self.best_val_acc, 100*self.best_acc,
                                                                    self.log_status, epoch+1, self.best_epoch,
                                                                            self.num_param) + self.file_name + '.txt'
        self.iterate_file = open(file_path,'w')
        self.iterate_file.write('Epoch  Train accuracy  Validation accuracy   Validation loss   epoch time(s)'
                                '   Test accuracy     Avg. F1     Min. F1    Min. class   test time(s)\n')
        for i in range(len(self.accuracy_table)):
            self.iterate_file.write('  {:3d}         {:.5f}              {:.5f}           {:.5f}            {:4.0f}'
                                    '         {:.5f}     {:.5f}     {:.5f}        {}              {:4.0f}\n'.
                                    format(i+1,self.accuracy_table[i][0],self.accuracy_table[i][1],
                                           self.accuracy_table[i][2],self.accuracy_table[i][3],
                                           self.accuracy_table[i][4],self.accuracy_table[i][5],
                                           self.accuracy_table[i][6],self.accuracy_table[i][7],
                                           self.accuracy_table[i][8]))
        self.iterate_file.close()
        if self.prev_file != None:
            os.remove(self.prev_file)
        self.prev_file = file_path


    def on_train_end(self, logs=None):
        finish_file_name = self.prev_file.replace('_WIP','_END')
        os.rename(self.prev_file, finish_file_name)

class ModifiedEarlyStopping(tf.keras.callbacks.Callback):
    # source: http://alexadam.ca/ml/2018/08/03/early-stopping.html

    def __init__(self,
                 monitors=['val_loss'],
                 min_deltas=[0],
                 minEpochs=1,
                 patience=0,
                 verbose=0,
                 modes=['auto'],
                 baselines=[None]
                ):
        super(ModifiedEarlyStopping, self).__init__()

        self.monitors = monitors
        self.baselines = baselines
        self.patience = patience
        self.verbose = verbose
        self.min_deltas = min_deltas
        self.minEpochs = minEpochs
        self.waits = []
        self.stopped_epoch = 0
        self.monitor_ops = []

        for i, mode in enumerate(modes):
            if mode not in ['auto', 'min', 'max']:
                warnings.warn('EarlyStopping mode %s is unknown, '
                              'fallback to auto mode.' % mode,
                              RuntimeWarning)
                modes[i] = 'auto'

        for i, mode in enumerate(modes):
            if mode == 'min':
                self.monitor_ops.append(np.less)
            elif mode == 'max':
                self.monitor_ops.append(np.greater)
            else:
                if 'accuracy' in self.monitors[i]:
                    self.monitor_ops.append(np.greater)
                else:
                    self.monitor_ops.append(np.less)

        # for i, monitor_op in enumerate(self.monitor_ops):
        #     if monitor_op == np.greater:
        #         self.min_deltas[i] *= 1
        #     else:
        #         self.min_deltas[i] *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.waits = []
        self.stopped_epoch = 0
        self.bests = []

        for i, baseline in enumerate(self.baselines):
            if baseline is not None:
                self.bests.append(baseline)
            else:
                self.bests.append(np.Inf if self.monitor_ops[i] == np.less else -np.Inf)

            self.waits.append(0)

    def on_epoch_end(self, epoch, logs=None):
        reset_all_waits = False
        for i, monitor in enumerate(self.monitors):
            current = logs.get(monitor)

            if current is None:
                warnings.warn(
                    'Early stopping conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (monitor, ','.join(list(logs.keys()))), RuntimeWarning
                )
                return

            if self.monitor_ops[i](current - self.min_deltas[i], self.bests[i]):
                self.bests[i] = current
                self.waits[i] = 0
                reset_all_waits = True
            else:
                if epoch > self.minEpochs:
                    self.waits[i] += 1

        if reset_all_waits:
            for i in range(len(self.waits)):
                self.waits[i] = 0

            return

        num_sat = 0
        for wait in self.waits:
            if wait >= self.patience:
                num_sat += 1

        if num_sat == len(self.waits):
            self.stopped_epoch = epoch
            self.model.stop_training = True

        print(self.waits)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

def doWork():
    global num_recurrent_features, selected_recurrent_features, num_annual_features, selected_annual_features, \
        num_static_features, selected_static_features, neighborhood, NC, starting_class, binaryFlag, resumeFlag, \
        inputdata_float16, float16_flag, remove_dup

    sess = Session(config=config)
    K.set_session(sess)
    strategy = tf.distribute.MirroredStrategy()
    local_device_protos = device_lib.list_local_devices()
    num_replicas = strategy.num_replicas_in_sync
    print('Device(s) available:{}\n'.format([x.name for x in local_device_protos]))
    train_batch_size = train_batch * num_replicas

    train_len = 0
    val_len = 0
    test_len = 0
    # counts = np.zeros(NC,)
    train_files = [glob.glob(inFolder + fp + 'config.txt') for fp in trainFiles]
    test_files = [glob.glob(inFolder + fp + 'test.gz') for fp in testFiles]
    train_files = sorted(item for sublist in train_files for item in sublist)
    test_files = sorted(item for sublist in test_files for item in sublist)
    train_files = [file[:-10] for file in train_files if file[0:len(inFolder) + 6] <= inFolder + lastBlock]  # file[:-11] is to strip the ending _config.txt from reported files
    test_files = [file for file in test_files if file[0:len(inFolder) + 6] <= inFolder + lastBlock]
    files_config = [file + 'config.txt' for file in train_files]
    files_train = [file + 'train.gz' for file in train_files]
    files_val = [file + 'val.gz' for file in train_files]
    files_test = test_files  # [file+'_test.gz' for file in files]
    numFiles = len(train_files)
    if numFiles == 0:
        print('No input data file found for file {}.\n'.format(trainFiles))
        return
    string = train_files[0][::-1]  # Get the first file and reverse its order
    string2 = string[string.find('(') + 2:]
    string3 = string2[:string2.find('_')]  # Extract the samplingScheme flag
    if 'B' in string3:  # files are generated for binary classification
        binaryFlag = True
        NC = 2
        starting_class = 1
    for file in files_config:
        with open(file, 'r') as f:
            contents = f.read()
            if binaryFlag:
                p1 = contents.find('class0')
                temp = contents[p1:]
                p2 = temp.find('=')
                p3 = temp.find('\n')
                binary_class0 = int(contents[p1 + p2 + 1:p1 + p3].strip())
            p1 = contents.find('remove duplicates = ')
            if p1 > 0:
                p2 = contents[p1+19:].find('\n')
                remove_dup = eval(contents[p1+19:p1+19+p2].strip())
            p1 = contents.find('float16_flag = ')
            if p1 > 0:
                p2 = contents[p1+14:].find('\n')
                inputdata_float16 = eval(contents[p1+14:p1+14+p2].strip())
                if inputdata_float16:
                    float16_flag = True
            p1 = contents.find('neighborhood')
            p2 = contents[p1:].find('=')
            p3 = contents[p1:].find('\n')
            neighborhood = int(contents[p1+p2+1:p1+p3].strip())
            p1 = contents.find('Number of training')
            temp = contents[p1:]
            p2 = temp.find('=')
            p3 = temp.find('\n')
            train_len += int(re.sub('[^0-9]', '', contents[p1 + p2 + 1:p1 + p3]))
            p1 = contents.find('Number of validation')
            temp = contents[p1:]
            p2 = temp.find('=')
            p3 = temp.find('\n')
            val_len += int(re.sub('[^0-9]', '', contents[p1 + p2 + 1:p1 + p3]))
            p1 = contents.find('Number of testing')
            temp = contents[p1:]
            p2 = temp.find('=')
            p3 = temp.find('\n')
            test_len += int(re.sub('[^0-9]', '', contents[p1 + p2 + 1:p1 + p3]))
            p1 = contents.find('Number of features')
            temp = contents[p1:]
            p2 = temp.find('=')
            temp = temp[p2 + 1:]
            p3 = temp.find(',')
            num_recurrent_features = int(temp[:p3].strip())
            temp = temp[p3 + 1:]
            p3 = temp.find(',')
            num_annual_features = int(temp[:p3].strip())
            temp = temp[p3 + 1:]
            p3 = temp.find('\n')
            num_static_features = int(temp[:p3].strip())
            p1 = contents.find('Included bands: ')
            temp = contents[p1 + 15:]
            p2 = temp.find(']')
            featureNames = eval(temp[:p2 + 1].strip())
            recurrent_feature_names = featureNames[:num_recurrent_features]
            annual_feature_names = featureNames[num_recurrent_features:num_recurrent_features + num_annual_features]
            static_feature_names = featureNames[num_recurrent_features + num_annual_features:]
            if selected_recurrent_features == 'all':
                selected_recurrent_features = np.arange(num_recurrent_features)
            if selected_annual_features == 'all':
                selected_annual_features = np.arange(num_annual_features)
            if selected_static_features == 'all':
                selected_static_features = np.arange(num_static_features)
            selected_recurrent_feature_names = [recurrent_feature_names[i] for i in selected_recurrent_features]
            selected_annual_feature_names = [annual_feature_names[i] for i in selected_annual_features]
            selected_static_feature_names = [static_feature_names[i] for i in selected_static_features]
            num_sel_features = len(selected_recurrent_features) + len(selected_annual_features) + len(selected_static_features)

    train_take = int(train_len / train_reduce)
    train_data = read_datasets(files_train, numFiles, type='train', numEpochs=max_epochs, batchSize=train_batch_size,
                               take=train_take, cache=cacheTrainDataset)
    val_take = int(val_len / val_reduce)
    val_data = read_datasets(files_val, numFiles, type='val', numEpochs=max_epochs, batchSize=train_batch_size,
                             take=val_take, cache=cacheValDataset)
    test_take = int(test_len / test_reduce)
    if extendedLogTest == 2:
        test_data = read_datasets(files_test, numFiles, type='test', numEpochs=max_epochs, batchSize=test_batchSize,
                                  take=test_take, cache=True)
    elif extendedLogTest == 1:
        test_data = read_datasets(files_test, numFiles, type='test', numEpochs=max_epochs, batchSize=test_batchSize,
                                  take=test_take, cache=False)
    else:
        test_data = read_datasets(files_test, numFiles, type='test', numEpochs=1, batchSize=test_batchSize,
                                  take=test_take, cache=False)

    train_steps = int(math.ceil(float(train_take) / train_batch_size))
    val_steps = int(math.ceil(float(val_take) / train_batch_size))
    test_steps = int(math.ceil(float(test_take) / test_batchSize))
    #it = train_data.make_one_shot_iterator()
    #t = sess.run(it.get_next())
    if mode == 'train':
        jobID = processID + 'LSTM' + str(RLayers).replace(' ', '') \
                + 'D1' + str(DLayers1).replace(' ', '') + 'D2' + str(DLayers2).replace(' ', '')
        if len(Dropouts) > 0:
            LSTM_drop = Dropouts[0]
            D1_drop = Dropouts[len(RLayers)]
            D2_drop = Dropouts[len(RLayers)+len(DLayers1)]
            jobID = jobID + 'Drop(' + str(LSTM_drop) + ',' + str(D1_drop) + ',' + str(D2_drop) + ')'
    else:
        jobID = processID + 'test_' + selected_model[:-3]

    logFile = outFolder + jobID + '.txt'
    if (os.path.exists(logFile)) and (resumeFlag != 'No'):
        log = open(logFile, 'a')
        log.write('\n *** Resume training ***:\n')
    else:
        log = open(logFile, 'w')

    startTick = time.time()
    if mode == 'train':
        if log_level == 0:
            log.write('Dataset(s) used:\n')
            log.write('\n'.join(' - {:}'.format(e) for e in train_files))
            log.write('\n')
        else:
            log.write('First dataset file is: {}\n'.format(train_files[0]))
        if binaryFlag:
            log.write('\nBinary class training with class0 code = {}'.format(binary_class0))
        log.write('\nRecurrent, 1st Dense, and 2nd Dense layers configuration: {}, {}, {}\n'.
                  format(RLayers, DLayers1, DLayers2))
        log.write('Dropout setting: {}\n'.format(Dropouts))
        log.write('Selected recurrent features: {}\n'.format(selected_recurrent_feature_names))
        log.write('Selected annual features: {}\n'.format(selected_annual_feature_names))
        log.write('Selected static features: {}\n'.format(selected_static_feature_names))
        log.write('Neighborhood window: {}\n'.format(neighborhood))
        log.write('Padded sequence length: {}\n'.format(maxSeqLen))
        log.write('Cache training dataset: {}, Cache validation dataset: {}\n'.format(cacheTrainDataset, cacheValDataset))
        log.write('float16_flag: {}, remove duplicates: {}\n'.format(float16_flag, remove_dup))
        log.write('Selected sensors: {}\n'.format(sensors))
        if reduceTime > 0:
            log.write('Reduce time type = {}, DOY values limited to values close to : {}\n'.format(reduceTime,selectDOY))
        log.write('Number of training, validation, and testing records: ({}, {}, {})\n'.format(train_len,val_len,test_len))
        log.write('Training, validation, and testing steps = ({}, {}, {})\n'.format(train_steps, val_steps, test_steps))
        log.write('Training mini-batch size: {}\n'.format(train_batch_size))
        log.write('Device(s) available:{}\n'.format([x.name for x in local_device_protos]))
        # Typical optimizer parameters:
        # 'SGD' optimizer_params = {'lr': 0.01, 'momentum': 0, 'decay':0, 'nesterov': False/True}
        # 'RMSprop' optimizer_params = {'lr': 0.001}
        # 'Adam' optimizer_params = {'lr': 0.001, 'amsgrad': False/True}
        # 'Nadam' optimizer_params = {'lr': 0.001}
        # 'Adagrad' optimizer_params = {'lr': 0.001}
        # 'Adadelta' optimizer_params = {'lr': 1.0}
        if optimizer_type == 'SGD':
            optimizer = tf.keras.optimizers.SGD(**optimizer_args)
        elif optimizer_type == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(**optimizer_args)
        elif optimizer_type == 'Adam':
            optimizer = tf.keras.optimizers.Adam(**optimizer_args)
        elif optimizer_type == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(**optimizer_args)
        elif optimizer_type == 'Adagrad':
            optimizer = tf.keras.optimizers.Adagrad(**optimizer_args)
        elif optimizer_type == 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(**optimizer_args)
        log.write('Optimizer: ' + optimizer_type + str(optimizer_args) + '\n')
        if kernel_reg:
            log.write('kernel_regularizer: {}\n'.format(kernel_reg))
        if recurrent_reg:
            log.write('recurrent_regularizer: {}\n'.format(recurrent_reg))
        log.write('Dense layer activation: {}\n'.format(dense_act))
        log.write('Minimum accuracy delta for early stopping: {}\n'.format(DELTAS))
        if num_replicas > 1:
            with strategy.scope():
                model = keras_model((None, len(selected_recurrent_features)), (len(selected_annual_features),),
                                    (len(selected_static_features),))
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['accuracy'])
        else:
            model = keras_model((None, len(selected_recurrent_features)), (len(selected_annual_features),),
                                (len(selected_static_features),))
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['accuracy'])
        best_model_file = outFolder + jobID + '_best_model.h5'
        last_model_file = outFolder + jobID + '_last_model.h5'
        # if a base model is given, load it.
        if os.path.exists(baseModelFolder + selected_model) and (resumeFlag != 'No'):
            print('-> loading from base model to resume.')
            log.write('Base model file:{}\n'.format(selected_model))
            if (resumeFlag == 'weights'):
                temp_model = load_model(baseModelFolder + selected_model)
                temp_model.save_weights(baseModelFolder + 'temp_model_weights.h5')
                model.load_weights(baseModelFolder + 'temp_model_weights.h5')
            else:  # load full model
                model = load_model(baseModelFolder + selected_model)
        # else if resume is requested and a previous last model exist, load it.
        elif os.path.exists(baseModelFolder + last_model_file) and (resumeFlag == 'last'):
            print('-> loading last saved model file to resume.')
            log.write('Base model file:{}\n'.format(last_model_file))
            model = load_model(last_model_file)
        else:  # Otherwise if resume flag is set wrongly, reverse it!
            resumeFlag = 'No'
        if log_level == 0:
            model.summary(print_fn=lambda x: log.write(x + '\n'))
        else:
            log.write('Model Total params: {}\n'.format(model.count_params()))
        model.summary()
        current_lr = K.get_value(model.optimizer.lr)
        print('Optimizer learning rate = {}'.format(current_lr))
        if not optimizer_args:
            log.write('Optimizer learning rate = {}\n'.format(current_lr))
        print('Number of training, validation, and testing records: ({}, {}, {})'.format(train_len, val_len, test_len))
        print('Training, validation, and testing steps = ({}, {}, {})\n'.format(train_steps, val_steps, test_steps))
        if resumeFlag != 'No':
            print('Calculating base model training accuracy...\n')
            base_train_loss, base_train_acc = model.evaluate(train_data, steps=train_steps, verbose=verbose)
            base_val_loss, base_val_acc = model.evaluate(val_data, steps=val_steps, verbose=verbose)
            print('Base training accuracy = {:.5f}, validation accuracy = {:.5f}\n'.format(base_train_acc, base_val_acc))
            log.write('Base training accuracy = {:.5f}, validation accuracy = {:.5f}\n'.format(base_train_acc, base_val_acc))
        else:
            base_val_acc = None
            base_val_loss = None

        mc = ModelCheckpoint(best_model_file, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        es = ModifiedEarlyStopping(monitors=['val_accuracy', 'val_loss'], modes=['max', 'min'],
                                   baselines=[base_val_acc, base_val_loss],
                                   min_deltas=DELTAS, verbose=1, minEpochs=min_epochs, patience=es_flag)
        mf = MakeFile(outFolder=outFolder, file_name=jobID, num_param=model.count_params(),
                      eval_data_iterator=iter(test_data), eval_steps=test_steps)
        # es = EarlyStopping(monitor='val_loss', min_delta = minDelta, verbose=1, patience=es_flag)
        CB = [mc]
        if createProgressFile:
            CB.append(mf)
        if es_flag > 0:
            CB.append(es)

        history = model.fit(train_data, epochs=max_epochs, steps_per_epoch=train_steps, validation_data=val_data,
                            validation_steps=val_steps, verbose=verbose, callbacks=CB)
        model.save(last_model_file, save_format='h5')
        trainingTick = time.time()
        log.write('Model training time was {:.0f} seconds (using {} GPUs)\n'.format(trainingTick - startTick, num_replicas))
        log.write('Max epochs: {}'.format(max_epochs))
        ep_end = len(history.history['loss'])
        if ep_end != max_epochs:
            log.write(', Early stopping at epoch {:d}'.format(ep_end))
        best_train_acc = np.max(history.history['accuracy'])
        best_train_val_acc = history.history['val_accuracy'][np.argmax(history.history['accuracy'])]
        best_epoch_train = np.argmax(history.history['accuracy']) + 1
        best_val_acc = np.max(history.history['val_accuracy'])
        best_epoch_val = np.argmax(history.history['val_accuracy']) + 1
        log.write('\nBest value of train accuracy was {:.5f} and at epoch {}, for which val. accuracy was {:.5f}.\n'.format(best_train_acc, best_epoch_train, best_train_val_acc))
        log.write('Best value of validation accuracy was {:.5f} and at epoch {}.\n'.format(best_val_acc, best_epoch_val))

        # Plot training and evaluation performance
        plt.figure()
        if dual_graph_output:
            plt.subplot(121)
            plt.plot(history.history['loss'], 'blue', label='Training')
            plt.plot(history.history['val_loss'], 'red', label='Validation')
            plt.grid(True)
            plt.legend()
            plt.title('Model performance')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.subplot(122)
            plt.plot(history.history['accuracy'], 'blue', label='Training')
            plt.plot(history.history['val_accuracy'], 'red', label='Validation')
            plt.grid(True)
            plt.legend()
            plt.title('Model performance')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
        else:
            plt.plot(history.history['loss'], 'blue', label='Training')
            plt.plot(history.history['val_loss'], 'red', label='Validation')
            plt.grid(True)
            plt.legend()
            plt.title('Model performance')
            plt.ylabel('loss')
            plt.xlabel('epoch')
        plt.savefig(outFolder + jobID + '.png')
        plt.close()
        model = load_model(best_model_file)
        if F_eval_set != None:
            log.write('\nNote: {} data partition is used for below evaluation.\n'.format(F_eval_set))
            log.write('\nBest model evaluation:\n')
            if F_eval_set == 'train':
                it = iter(train_data)
                eval_steps = train_steps
            elif F_eval_set == 'test':
                it = iter(test_data)
                eval_steps = test_steps
            else:
                it = iter(val_data)
                eval_steps = val_steps
            y_eval, y_pred, y_pred_prob, seq_len = calculate_predictions(model, it, eval_steps)
            confusion = confusion_matrix(y_eval, y_pred, labels=np.arange(NC))
            report = classification_report(y_eval, y_pred, digits=5, output_dict=True)
            np.savez_compressed(outFolder + jobID + '_predictions.npz', y_eval=y_eval, y_pred_prob=y_pred_prob,
                                seq_len=seq_len, report=report, confusion=confusion)
            if log_level == 0:
                log.write('\n\nEvaluation set confusion matrix:\n{}\n'.format(confusion))
            unwanted = {'accuracy', 'macro avg', 'weighted avg'}
            for unwanted_key in unwanted: del report[unwanted_key]
            f1_list = [report[key]['f1-score'] for key in report.keys()]
            f1_min_index = f1_list.index(min(f1_list))
            log.write('\nEvaluation set classification report:\n{}'.format(classification_report(y_eval, y_pred, digits=5)))
            log.write('\nMinimum F1 value: {:.5f} at class {}\n'.format(f1_list[f1_min_index], list(report.keys())[f1_min_index]))
            # AUCs, plots = calculate_AUC(y_eval_onehot, y_pred_scores, NC)
            # #plots.savefig(outFolder + jobID + '_best_ROC.png')
            # plots.close()
            # keys = AUCs.keys()
            # s = ' '.join('{}:{:.2f}, '.format(e, AUCs[e]) for e in keys)
            # log.write('\nArea Under Curve for each class, macro, and micro: {}\n'.format(s[:-2]))
            # log.write('\nLog and evaluation time = {:.0f} seconds\n'.format(time.time() - finishTick))
        testingTick = time.time()
        log.write('Model testing time was {:.0f} seconds\n'.format(testingTick - trainingTick))
    else:
        log.write('Testing model file:{}\n'.format(selected_model))
        if log_level == 0:
            log.write('Dataset(s) used:\n')
            log.write('\n'.join(' - {:}'.format(e) for e in test_files))
            log.write('\n')
        else:
            log.write('First dataset file is: {}\n'.format(test_files[0]))
        log.write('Selected recurrent features: {}\n'.format(selected_recurrent_feature_names))
        log.write('Selected annual features: {}\n'.format(selected_annual_feature_names))
        log.write('Selected static features: {}\n'.format(selected_static_feature_names))
        log.write('Neighborhood window: {}\n'.format(neighborhood))
        log.write('float16_flag: {}\n'.format(float16_flag))
        log.write('Selected sensors: {}\n'.format(sensors))
        if reduceTime > 0:
            log.write('Reduce time type = {}, DOY values limited to values close to : {}\n'.format(reduceTime,selectDOY))
        log.write('Number of testing records: {}, testing steps: {})\n'.format(test_len, test_steps))
        model = load_model(baseModelFolder + selected_model)
        print('Evaluating the input model {}'.format(selected_model))
        log.write('Dataset(s) used:\n')
        if F_eval_set == 'train':
            log.write('\n'.join(' - {:}'.format(e) for e in train_files))
            it = iter(train_data)
            eval_steps = train_steps
        elif F_eval_set == 'test':
            log.write('\n'.join(' - {:}'.format(e) for e in test_files))
            it = iter(test_data)
            eval_steps = test_steps
        else:
            log.write('\n'.join(' - {:}'.format(e) for e in train_files))
            it = iter(val_data)
            eval_steps = val_steps
        y_eval, y_pred, y_pred_prob, seq_len = calculate_predictions(model, it, eval_steps)
        confusion = confusion_matrix(y_eval, y_pred, labels=np.arange(NC))
        report = classification_report(y_eval, y_pred, digits=5, output_dict=True)
        np.savez_compressed(outFolder + jobID + '_predictions.npz', y_eval=y_eval, y_pred_prob=y_pred_prob,
                            seq_len=seq_len, report=report, confusion=confusion)
        if log_level == 0:
            log.write('\n\nEvaluation set confusion matrix:\n{}\n'.format(confusion))
        unwanted = {'accuracy', 'macro avg', 'weighted avg'}
        for unwanted_key in unwanted: del report[unwanted_key]
        f1_list = [report[key]['f1-score'] for key in report.keys()]
        f1_min_index = f1_list.index(min(f1_list))
        log.write('\nEvaluation set classification report:\n{}'.format(classification_report(y_eval, y_pred, digits=5)))
        log.write('\nMinimum F1 value: {:.5f} at class {}\n'.format(f1_list[f1_min_index], list(report.keys())[f1_min_index]))
        #AUCs, plots = calculate_AUC(y_eval_onehot, y_pred_scores, NC)
        #plots.savefig(outFolder + jobID + '_ROC.png')
        #plots.close()
        #keys = AUCs.keys()
        #s = ' '.join('{}:{:.2f}, '.format(e, AUCs[e]) for e in keys)
        #log.write('\nArea Under Curve for each class, macro, and micro: {}\n'.format(s[:-2]))
        testingTick = time.time()
        log.write('Model testing time was {:.0f} seconds\n'.format(testingTick - startTick))
    log.close()
    K.clear_session()
    sess.close()
    print('========== Process {} ended successfully =========='.format(processID))

if __name__ == '__main__':
    startTime = time.time()
    # If a list of models is given, test features should match it. Set_no should also match it but it will be matched later.
    if len(baseModels) > 0:
        testFeatures = []
        for i in range(len(baseModels)):
            model = baseModels[i]
            temp = model[model.find(']') + 2:]
            feature = temp[:temp.find('_')]
            if feature == 'ATE':
                feature = 1000
            elif feature == 'APR':
                feature = 1001
            elif feature == 'ECO':
                feature = 1100
            elif feature == 'NTM':
                feature = 1200
            elif feature == 'NTS':
                feature = 1201
            elif feature == 'NPM':
                feature = 1202
            elif feature == 'NPS':
                feature = 1203
            else:
                feature = int(feature)
            testFeatures.append(feature)

    #NumProc = 2
    #pool = Pool(processes=NumProc)
    #pool.map(doSimulation, (0,1))
    for index in range(len(testFeatures)):
        feature = testFeatures[index]
        if len(baseModels) > 0:  # if base models are supplied, set number should be extracted from them and override other setting
            selected_model = baseModels[index]
            set_no = [int(selected_model[selected_model.find('[') + 1:selected_model.find(']')])]
        for i in set_no:
            trainFiles = [x + '_' + str(i) + '_*' for x in trainFilePattern]
            testFiles = testFilePattern
            pID = pid_prefix + '[' + str(i) + ']'
            if feature == 0:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features
                selected_static_features = fixed_static_features
                processID = pID + '_000_'
            elif (feature < 1000) and (feature not in fixed_recurrent_features):
                selected_recurrent_features = fixed_recurrent_features + [feature]
                selected_annual_features = fixed_annual_features
                selected_static_features = fixed_static_features
                processID = pID + '_' + str(feature).zfill(3) + '_'
            elif feature == 1000:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features + np.arange(12).tolist()
                selected_static_features = fixed_static_features
                processID = pID + '_ATE_'
            elif feature == 1001:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features + np.arange(12, 24).tolist()
                selected_static_features = fixed_static_features
                processID = pID + '_APR_'
            elif feature == 1100:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features
                selected_static_features = [0] + fixed_static_features
                processID = pID + '_ECO_'
            elif feature == 1200:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features
                selected_static_features = fixed_static_features + np.arange(3, 15).tolist()
                processID = pID + '_NTM_'
            elif feature == 1201:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features
                selected_static_features = fixed_static_features + [15, 16, 17]
                processID = pID + '_NTS_'
            elif feature == 1202:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features
                selected_static_features = fixed_static_features + np.arange(18, 30).tolist()
                processID = pID + '_NPM_'
            elif feature == 1203:
                selected_recurrent_features = fixed_recurrent_features
                selected_annual_features = fixed_annual_features
                selected_static_features = fixed_static_features + [30, 31, 32]
                processID = pID + '_NPS_'
            random.seed()
            #doWork()
            p = multiprocessing.Process(target=doWork)
            p.start()
            p.join()

    print('The whole job took {:.0f} seconds to complete.'.format(time.time() - startTime))

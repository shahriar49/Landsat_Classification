#######################################################################################################################
# Specialized script to test a model on all blocks, based on NASA_RNN_V8_4.
# Update V2.0: Add the possibility to filter by Landsat sensor or DOY.
########################################################################################################################
# Shahriar S. Heydari, 12/20/2020

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: no INFO, 2: no INFO and WARNING, 3: no INFO, WARNING, and ERROR printed
import numpy as np
import math, time, glob, random, re
from sklearn.metrics import classification_report, confusion_matrix

# Important parameters to set before execution
inFolder = '/home/sshahhey/TFRecord/allblocks2/'
outFolder = '/home/sshahhey/WIP/'
baseModelFolder = '/home/sshahhey/Models/'
selected_model = 'FH_0[0]_000_LSTM(320,320,320)D1(64,32)D2(256,256,256)Drop(0.25,0.1,0)_best_model.h5'
pid_suffix = '_LS7'
set_no = 0
gpu_no = '0'
sensors = [.5,.7,.8]
reduceTime = 0      # if 0, full year sdequences are used,
                    # If 1, the time range between the first and second elements of selectDOY are used,
#                   # if 2, the observation closest to selectDOY elements are used.
selectDOY = [1,180]  # (15, 106, 197, 288 for 4 seasons)
selectDOYs = np.array(selectDOY)/366
selected_recurrent_features = [0, 1, 2, 3, 4, 5, 6, 7, 10, 17, 37, 30]
selected_annual_features = []
selected_static_features = [0, 1, 2]
trainFilePattern = ['*']
testFilePattern = ['*']
float16_flag = False
log_level = 0  # 0 = All information is logged, 1= Important information is logged

# Other variables that are not changed typically
neighborhood = 1
NC = 7  # Default number of allowed landcover classes (from 21 to 27)
starting_class = 21  # Default land cover code of the first class
lastBlock = 'samp84'
maxSeqLen = 100  # Maximum length of input sequences over all blocks
test_batchSize = 1024
num_recurrent_features = 0
num_sel_features = len(selected_recurrent_features) + len(selected_annual_features) + len(selected_static_features)

# Setup Tensorflow and its related modules
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: no INFO, 2: no INFO and WARNING, 3: no INFO, WARNING, and ERROR printed
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
config = ConfigProto()
config.gpu_options.allow_growth = True

# Input TFRecord dataset structure definition dictionary
featuresDict = {'data': tf.io.FixedLenFeature([], dtype=tf.string),
                'annualData': tf.io.FixedLenFeature([], dtype=tf.string),
                'staticData': tf.io.FixedLenFeature([], dtype=tf.string),
                'rows': tf.io.FixedLenFeature([], dtype=tf.int64),
                'label': tf.io.FixedLenFeature([], dtype=tf.int64)
                }

def parse_tfrecord(example):
    def closest(lst, K):
        idx = tf.compat.v1.arg_min(tf.abs(lst - K),0)
        return idx

    features = tf.io.parse_single_example(serialized=example, features=featuresDict)
    label = tf.one_hot(features['label'] - starting_class, NC)
    rows = features['rows']
    recurrentData = tf.io.decode_raw(features['data'], tf.float32)
    if float16_flag:
        recurrentData = tf.cast(recurrentData, tf.float16)
    recurrentData = tf.reshape(recurrentData, (rows, num_recurrent_features))
    recurrentData = tf.gather(recurrentData, selected_recurrent_features, axis=1)
    sensor_list = recurrentData[:,1]
    mask = tf.greater(sensor_list,1.0)
    if 0.5 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.5))
    if 0.7 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.7))
    if 0.8 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.8))
    recurrentData = tf.boolean_mask(recurrentData, mask)
    if reduceTime == 2:
        index = [closest(recurrentData[:, 0], sd) for sd in selectDOYs]
        recurrentData = tf.gather(recurrentData, index, axis=0)
    elif reduceTime == 1:
        start = tf.greater_equal(recurrentData[:, 0], selectDOYs[0])
        end = tf.greater_equal(tf.negative(recurrentData[:, 0]), -selectDOYs[1])
        recurrentData = tf.boolean_mask(recurrentData, tf.logical_and(start, end))
    dataLen = tf.shape(input=recurrentData)[0]
    recurrentData = tf.pad(tensor=recurrentData, paddings=[[maxSeqLen - dataLen, 0], [0, 0]])
    if selected_annual_features:
        annualData = tf.gather(tf.io.decode_raw(features['annualData'], tf.float32), selected_annual_features)
        if float16_flag:
            annualData = tf.cast(annualData, tf.float16)
    else:
        annualData = []
    if selected_static_features:
        staticData = tf.gather(tf.io.decode_raw(features['staticData'], tf.float32), selected_static_features)
        if float16_flag:
            staticData = tf.cast(staticData, tf.float16)
    else:
        staticData = []
    return {'recurrentData': recurrentData, 'annualData': annualData, 'staticData': staticData}, label

def read_datasets(pattern, numFiles, numEpochs=None, batchSize=None):
    files = tf.data.Dataset.list_files(pattern)

    def _parse(x):
        x = tf.data.TFRecordDataset(x, compression_type='GZIP')
        return x

    dataset = files.interleave(_parse, cycle_length=numFiles, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(lambda x: parse_tfrecord(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchSize)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(numEpochs)
    return dataset

def calculate_test_statistics(model, data_iterator, eval_steps):

    #sess = tf.keras.backend.get_session()
    #next_element = data_iterator.get_next()
    y_eval = y_pred = np.empty(0)
    for i in range(eval_steps):
        if i % 1000 == 0:
            print(' - Evaluating {} of {} records done'.format(i, eval_steps))
        batch = next(data_iterator)
        x_eval_batch = batch[0]
        y_eval_batch = batch[1]
        y_pred_batch = model.predict_on_batch(x_eval_batch)
        # y_eval_onehot = np.append(y_eval_onehot, y_eval_batch, axis=0)
        # y_pred_scores = np.append(y_pred_scores, y_pred_batch, axis=0)
        y_eval = np.append(y_eval, np.argmax(y_eval_batch, axis=1))
        y_pred = np.append(y_pred, np.argmax(y_pred_batch, axis=1))
    return y_eval, y_pred

def doWork():
    global num_recurrent_features

    sess = Session(config=config)
    K.set_session(sess)

    test_len = 0
    # counts = np.zeros(NC,)
    config_file = [glob.glob(inFolder + fp + 'config.txt') for fp in trainFiles][0][0]
    test_files = [glob.glob(inFolder + fp + 'test.gz') for fp in testFiles][0]
    with open(config_file, 'r') as f:
        contents = f.read()
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

    test_data = read_datasets(test_files, 1, numEpochs=1, batchSize=test_batchSize)
    test_steps = int(math.ceil(float(test_len) / test_batchSize))
    #it = train_data.make_one_shot_iterator()
    #t = sess.run(it.get_next())
    jobID = processID + 'test_' + selected_model[:-3]
    logFile = outFolder + jobID + '.txt'
    log = open(logFile, 'w')

    startTick = time.time()
    model = load_model(baseModelFolder + selected_model)
    log.write('Testing model: {}\n'.format(selected_model))
    print('Testing model: {}'.format(selected_model))
    log.write('Number of input bands: {}, neighborhood window: {}, float16: {}\n'.format(num_sel_features,
                                                                                         neighborhood, float16_flag))
    log.write('Selected Landsat sensors: {}\n'.format(sensors))
    if reduceTime > 0:
        log.write('Reduce_time flag: {}, selectDOY = {}\n'.format(reduceTime, selectDOY))
    log.write('Dataset(s) used:\n')
    log.write('\n'.join(' - {:}'.format(e) for e in test_files))
    it = iter(test_data)
    y_eval, y_pred = calculate_test_statistics(model, it, test_steps)
    log.write('\n\nEvaluation set confusion matrix:\n{}\n'.format(confusion_matrix(y_eval, y_pred, labels=np.arange(NC))))
    report = classification_report(y_eval, y_pred, digits=5, output_dict=True)
    result_accuracy = report['accuracy']
    unwanted = set(['accuracy', 'macro avg', 'weighted avg'])
    for unwanted_key in unwanted: del report[unwanted_key]
    f1_list = [report[key]['f1-score'] for key in report.keys()]
    support_list = [report[key]['support'] for key in report.keys()]
    f1_min_index = f1_list.index(min(f1_list))
    f1_min = f1_list[f1_min_index]
    f1_mean = sum(f1_list)/len(f1_list)
    classes = ['0.0','1.0','2.0','3.0','4.0','5.0','6.0']
    missing = [x for x in classes if not x in report.keys()]
    for c in range(len(missing)):
        f1_list = f1_list + [-1]
        support_list = support_list + [0]
    classes, f1_list, support_list = zip(*sorted(zip(list(report.keys())+missing, f1_list, support_list)))
    results = {'accuracy':result_accuracy, 'avg_F1': f1_mean, 'min_F1': f1_min,'support_list':support_list,
               'f1_list':f1_list,'f1_min_index': f1_min_index}
    log.write('\nEvaluation set classification report:\n{}'.format(classification_report(y_eval, y_pred, digits=5)))
    log.write('\nMinimum F1 value: {:.5f} at class {}\n'.format(f1_min, list(report.keys())[f1_min_index]))
    log.write('Model testing time was {:.0f} seconds\n'.format(time.time() - startTick))
    log.close()
    K.clear_session()
    sess.close()
    print('========== Process {} ended successfully =========='.format(processID))
    return results

if __name__ == '__main__':
    startTime = time.time()
    # test_features is assumed to be 000 (no additional feature is placed to those in the fixed features setting
    summary_file = open(outFolder+'summary_results'+pid_suffix+'.csv','w')
    summary_file.write('Block,Accuracy,Average_F1,Minimum_F1,Min_class,Water_sup,Water_F1,Imp_sup,Imp_F1,Grass_sup,'
                       'Grass_F1,Forest_sup,Forest_F1,Bare_sup,Bare_F1,Agri_sup,Agri_F1,Wetland_sup,Wetland_F1\n')
    for i in range(1,2):
        samp_id = 'samp' + str(i).zfill(2)
        trainFiles = [samp_id + '*_' + str(set_no) + '_*']
        testFiles = [samp_id + '*']
        processID = samp_id + pid_suffix + '_'
        random.seed()
        result = doWork()
        summary_file.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(samp_id,
                            result['accuracy'], result['avg_F1'],
                            result['min_F1'], result['f1_min_index'],
                            result['support_list'][0],result['f1_list'][0],
                            result['support_list'][1],result['f1_list'][1],
                            result['support_list'][2], result['f1_list'][2],
                            result['support_list'][3], result['f1_list'][3],
                            result['support_list'][4], result['f1_list'][4],
                            result['support_list'][5], result['f1_list'][5],
                            result['support_list'][6], result['f1_list'][6]))

    summary_file.close()
    print('The whole job took {:.0f} seconds to complete.'.format(time.time() - startTime))

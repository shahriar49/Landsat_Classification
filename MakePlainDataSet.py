#######################################################################################################################
# This script read allblock2 dataset and creates plain datasets for RF and MLP classifiers
#######################################################################################################################
# Shahriar S. Heydari, 12/29/2020

import numpy as np
import math, time, glob, random, re, os #,psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: default, 1: no INFO, 2: no INFO and WARNING, 3: no INFO, WARNING, and ERROR printed
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.compat.v1 import ConfigProto
config = ConfigProto()
config.gpu_options.allow_growth = True
#sess = tf.compat.v1.InteractiveSession(config=config)

# Default parameter values (will be overridden by set_parameters function if input configuration file is provided)
inFolder = '/home/sshahhey/TFRecord/allblocks2/'
outFolder = '/home/sshahhey/WIP/plain_datasets/'
NPZfileName = 'allblocks2_plain_full_random_16M'
set_nos = [0]
pid_prefix = NPZfileName
trainFilePattern = ['*']
testFilePattern = ['*']
selected_recurrent_features = np.arange(44).tolist()#[6,10,20,7,8,3,11,9,5,21,2,4,0,36,17,30,37]
selected_annual_features = np.arange(24).tolist()#[]
selected_static_features = np.arange(33).tolist()#[0,1,27,32,25,19,31,28,26,24,18,29,23,22,20,21,30]
input_batch = 4096
max_len = 100
train_reduce = 1
val_reduce = 1
test_reduce = 1
float16_flag = False
sensors = [.5,.7,.8]
reduceTime = 3      # if 0, full year sequences are used,
                    # If 1, the time range between the first and second elements of selectDOY are used,
                    # if 2, the observation closest to selectDOY elements are used,
                    # if 3, one random date from each sequence is selected
selectDOY = [0,180]  # (15, 106, 197, 288 for 4 seasons)
selectDOYs = np.array(selectDOY)/366
lastBlock = 'samp84'
#numCores = psutil.cpu_count()
#NumProc = 1
starting_class = 21

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

    features = tf.io.parse_single_example(example, featuresDict)
    label = tf.cast(features['label'] - starting_class, tf.float32)
    if float16_flag:
        label = tf.cast(label, tf.float16)
    rows = features['rows']
    Data = tf.io.decode_raw(features['data'], tf.float32)
    if float16_flag:
        Data = tf.cast(Data, tf.float16)
    Data = tf.reshape(Data, (rows, num_recurrent_features))
    Data = tf.gather(Data, selected_recurrent_features, axis=1)
    sensor_list = Data[:,1]
    mask = tf.greater(sensor_list,1.0)
    if 0.5 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.5))
    if 0.7 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.7))
    if 0.8 in sensors:
        mask = tf.logical_or(mask, tf.equal(sensor_list,0.8))
    Data = tf.boolean_mask(Data, mask)
    if reduceTime == 1:
        start = tf.greater_equal(Data[:, 0], selectDOYs[0])
        end = tf.greater_equal(tf.negative(Data[:, 0]), -selectDOYs[1])
        Data = tf.boolean_mask(Data, tf.logical_and(start, end))
    elif reduceTime == 2:
        index = [closest(Data[:, 0], sd) for sd in selectDOYs]
        Data = tf.gather(Data, index, axis=0)
    elif reduceTime == 3:
        pick = tf.random.uniform(shape=(), minval=0, maxval=tf.shape(Data)[0], dtype=tf.int32)
        Data = tf.expand_dims(Data[pick,:], axis=0)
    l = tf.shape(Data)[0]
    if selected_annual_features:
        annualData = tf.gather(tf.io.decode_raw(features['annualData'], tf.float32), selected_annual_features)
        #annualData = tf.io.decode_raw(features['annualData'], tf.float32)
        if float16_flag:
            annualData = tf.cast(annualData, tf.float16)
        Data = tf.concat([Data, tf.tile(tf.expand_dims(annualData, axis=0), [l,1])], axis=1)
    if selected_static_features:
        staticData = tf.gather(tf.io.decode_raw(features['staticData'], tf.float32), selected_static_features)
        #staticData = tf.io.decode_raw(features['staticData'], tf.float32)
        if float16_flag:
            staticData = tf.cast(staticData, tf.float16)
        Data = tf.concat([Data, tf.tile(tf.expand_dims(staticData, axis=0), [l,1])], axis=1)
    Data = tf.concat([Data, tf.tile(tf.expand_dims(tf.expand_dims(label, axis=0),axis=1), [l,1])], axis=1)
    Data = tf.pad(tensor=Data, paddings=[[max_len - l, 0], [0, 0]])
    return Data

def read_datasets(pattern, numFiles, numEpochs=None, batchSize=None, take=1, cache=False):
    files = tf.data.Dataset.list_files(pattern)

    def _parse(x):
        x = tf.data.TFRecordDataset(x, compression_type='GZIP')
        return x

    #np = int(numCores / NumProc)
    dataset = files.interleave(_parse, cycle_length=numFiles, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(lambda x: parse_tfrecord(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.take(take)
    dataset = dataset.batch(batchSize)
    if cache:
        print('cache enabled for take=',take)
        dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(numEpochs)
    return dataset

def doWork():
    global num_recurrent_features, selected_recurrent_features, num_annual_features, selected_annual_features, \
        num_static_features, selected_static_features, neighborhood, NC, starting_class, binaryFlag, resumeFlag, \
        float16_flag, remove_dup, reduceTime, selectDOY, input_batch

    #sess = tf.Session()

    num_train_seq = 0
    num_val_seq = 0
    num_test_seq = 0
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
    for file in files_config:
        with open(file, 'r') as f:
            contents = f.read()
            p1 = contents.find('neighborhood')
            p2 = contents[p1:].find('=')
            p3 = contents[p1:].find('\n')
            neighborhood = int(contents[p1+p2+1:p1+p3].strip())
            p1 = contents.find('Number of training')
            temp = contents[p1:]
            p2 = temp.find('=')
            p3 = temp.find('\n')
            num_train_seq += int(re.sub('[^0-9]', '', contents[p1 + p2 + 1:p1 + p3]))
            p1 = contents.find('Number of validation')
            temp = contents[p1:]
            p2 = temp.find('=')
            p3 = temp.find('\n')
            num_val_seq += int(re.sub('[^0-9]', '', contents[p1 + p2 + 1:p1 + p3]))
            p1 = contents.find('Number of testing')
            temp = contents[p1:]
            p2 = temp.find('=')
            p3 = temp.find('\n')
            num_test_seq += int(re.sub('[^0-9]', '', contents[p1 + p2 + 1:p1 + p3]))
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

    train_take = int(num_train_seq / train_reduce)
    train_data = read_datasets(files_train, numFiles, numEpochs=1, batchSize=input_batch, take=train_take)
    val_take = int(num_val_seq / val_reduce)
    val_data = read_datasets(files_val, numFiles, numEpochs=1, batchSize=input_batch, take=val_take)
    test_take = int(num_test_seq / test_reduce)
    test_data = read_datasets(files_test, numFiles, numEpochs=1, batchSize=input_batch, take=test_take)
    train_steps = int(math.ceil(float(train_take) / input_batch))
    val_steps = int(math.ceil(float(val_take) / input_batch))
    test_steps = int(math.ceil(float(test_take) / input_batch))

    print('Train_take = {}, val_take = {}, test_take = {}'.format(train_take, val_take, test_take))
    X_train = X_val = X_test = np.zeros((0,num_sel_features))
    Y_train = Y_val = Y_test = np.zeros((0,))
    num_train_rec = num_val_rec = num_test_rec = 0
    it = iter(train_data)#.make_one_shot_iterator()
    print('Train steps = {}'.format(train_steps))
    for i in range(train_steps):
        t = next(it).numpy()#sess.run(it.get_next())
        rec = t.reshape(-1,num_sel_features+1)
        rec = rec[~np.all(rec == 0, axis=1)]
        l = len(rec)
        num_train_rec += l
        X_train = np.vstack((X_train, rec[:,:-1]))
        Y_train = np.hstack((Y_train, rec[:,-1]))
        if (i % 100 == 0):
            print('  {} batches read.'.format(i+1))
    indices = np.arange(num_train_rec)
    np.random.shuffle(indices)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    print('Val steps = {}'.format(val_steps))
    it = iter(val_data)#.make_one_shot_iterator()
    for i in range(val_steps):
        t = next(it).numpy()#sess.run(it.get_next())
        rec = t.reshape(-1,num_sel_features+1)
        rec = rec[~np.all(rec == 0, axis=1)]
        l = len(rec)
        num_val_rec += l
        X_val = np.vstack((X_val, rec[:,:-1]))
        Y_val = np.hstack((Y_val, rec[:,-1]))
        if (i % 100 == 0):
            print('  {} batches read.'.format(i+1))
    print('Test steps = {}'.format(test_steps))
    it = iter(test_data)#.make_one_shot_iterator()
    for i in range(test_steps):
        t = next(it).numpy()#sess.run(it.get_next())
        rec = t.reshape(-1,num_sel_features+1)
        rec = rec[~np.all(rec == 0, axis=1)]
        l = len(rec)
        num_test_rec += l
        X_test = np.vstack((X_test, rec[:,:-1]))
        Y_test = np.hstack((Y_test, rec[:,-1]))
        if (i % 100 == 0):
            print('  {} batches read.'.format(i+1))
    outputFile = outFolder + NPZfileName + '[' + str(set_no) + ']' + '.npz'
    np.savez_compressed(outputFile, X_train=X_train, Y_train=Y_train, X_val=X_val,
                        Y_val=Y_val, X_test=X_test, Y_test=Y_test,
                        selected_recurrent_feature_names=selected_recurrent_feature_names,
                        selected_annual_feature_names=selected_annual_feature_names,
                        selected_static_feature_names=selected_static_feature_names,
                        neighborhood=neighborhood, float16_flag=float16_flag,
                        num_train_seq=num_train_seq, num_val_seq=num_val_seq, num_test_seq=num_test_seq,
                        num_train_rec=num_train_rec, num_val_rec=num_val_rec, num_test_rec=num_test_rec,
                        reduceTime=reduceTime, selectDOY=selectDOY)

    log = open(outFolder + NPZfileName + '[' + str(set_no) + ']' + '.txt', 'w')
    log.write('First dataset file is: {}\n'.format(train_files[0]))
    log.write('Selected recurrent features: {}\n'.format(selected_recurrent_feature_names))
    log.write('Selected annual features: {}\n'.format(selected_annual_feature_names))
    log.write('Selected static features: {}\n'.format(selected_static_feature_names))
    log.write('Neighborhood window: {}\n'.format(neighborhood))
    log.write('float16_flag: {}\n'.format(float16_flag))
    log.write('Number of training, validation, and testing sequences: ({}, {}, {})\n'
              .format(num_train_seq, num_val_seq, num_test_seq))
    log.write('Number of selected training, validation, and testing records = ({}, {}, {})\n'
              .format(num_train_rec, num_val_rec, num_test_rec))
    if reduceTime > 0:
        log.write('reduceTime flag = {}'.format(reduceTime))
        if reduceTime < 3:
            log.write(' DOY selection criteria : {}\n'.format(selectDOY))
        else:
            log.write('\n')
    log.close()
    print('Dataset saved to {}'.format(outFolder + NPZfileName+ '[' + str(set_no) + ']' + '.npz'))

if __name__ == '__main__':
    startTime = time.time()
    for set_no in set_nos:
        trainFiles = [x + '_' + str(set_no) + '_*' for x in trainFilePattern]
        testFiles = testFilePattern
        random.seed()
        doWork()
    print('The whole job took {:.0f} seconds to complete.'.format(time.time() - startTime))

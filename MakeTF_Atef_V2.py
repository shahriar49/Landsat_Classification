###############################################################################################################
# Update V6.5:
#       - Add possibility of including elevation band in CNN data
###############################################################################################################
# Shahriar S. Heydari, 1/12/2021

import tensorflow as tf
import numpy as np
import os, time, gc
import fnmatch
from sklearn.model_selection import train_test_split

inFolder = 'D:\\Shahriar\\temp\\6\\'
outFolder = 'D:\\Shahriar\\temp\\6\\output\\'
files = sorted(os.listdir(inFolder))
files = fnmatch.filter(files, '*.npz')
repeat = (0,)  # (0,1,2,3,4,5,6,7)  # number of datasets that will be generated with the same train/test data sizes
samplingScheme = 'stratified'
# 'equal': equal number of samples are taken from each class
# 'stratified': sampling is stratified
# 'random': sampling is random
# 'fixed': total number of samples per class and their trainig/validation/test ratios are given by user

if samplingScheme == 'fixed':
    classDistribution = np.array((100,200,300,400))
    ratios = (0.5,0.1,0.3)           # train/validation/test ratio. Should be float but not necessarily sum to 1.0
    totalPoints = sum(classDistribution)
    trainClassFreqs = np.floor(classDistribution * ratios[0]).astype(int)
    valClassFreqs = np.floor(classDistribution * ratios[1]).astype(int)
    testClassFreqs = np.floor(classDistribution * ratios[2]).astype(int)
    data_sizes = (sum(trainClassFreqs), sum(valClassFreqs), sum(testClassFreqs))
else:
    data_sizes = (200,20,40)
# data_sizes is the tuple for number of train, validation, and test  points or their ratio from the total samples.
# If sampling scheme is 'equal', the specified number is taken for each class (therefore the total number of samples
# will be these numbers times number of classes). If sampling scheme is 'stratified' or 'random', the specified
# number is taken for the whole classes and distributed stratified or random between classes.
# If number of train points are zero, then all remaining points (after picking test and
# validation points) are assigned for training

NC = 7                          # Number of allowed landcover classes (from 21 to 27 in our case)
starting_class = 21             # Land cover code of the first class
binaryClass0 = None             # if None, original classes are preserved. If a number, the given number will be class#1
                                # and all other classes will be aggregated to class#2
neighborhood = 3                # should be an odd number
LSbands = ['blue','green','red','NIR','SWIR1','SWIR2']      # Name of Landsat spectral bands
CNNbands = ['blue','green','red','NIR','SWIR1','SWIR2']     # Name of bands to be used for generating spatial data
addElevation2CNN = True
reuseTestPoints = False
reuseTrainPoints = False
remove_dups = False             # if true, duplicate DOY entries are removed
maskCloud = True                # if true, do cloud and cloud shadow filtering based on QA bits

yearRange = (2005,2019)         # Both start and end years are included in the acceptable range
years2pickRule = 'all'           # 'all' (all years are taken), 'fixed' (years are specified),
                                # 'systematic' (years are evenly spaced), 'random' (random number of years are taken)
# Years to sample are specified in below lists if the rule is not 'all'. In the case of 'systematic' or 'random' rule,
# only the number of desired years are given and not the list of year numbers.
years2pick_train = None         #3  #[2010,2011]
years2pick_test = None          #3  #[2011,2012]
reduceTime = False              # if true, only the time stamps closest to seasonDOY vector are kept in each sequence
seasonDOYs = [15, 106, 197, 288]  # Approximate mid season DOYs
# Maximum values for each band. The band values are normalized to the range of 0 ~ 2.
DOY_max = 366.0
sensor_max = 10.0
ECOID_max = 850.0
elevation_min = -20         # meter
elevation_max = 6500.0      # meter
slope_max = 360.0           # degree
aspect_max = 360.0          # degree
DD_max = 2.0
GLCM_diss_max = 100.0
GLCM_ent_max = 10.0
GLCM_svag_max = 250.0
GLCM_var_max = 1500.0
temp_min = -30.0            # degree centigrade
temp_max = 50.0             # degree centigrade
rain_max = 2650.0           # mm
Nrain_max = 12000.0         # mm

# Below variables are assigned later.
bands = []
divide_to = []
add_to = []
sBands = []
divide_to_s = []
add_to_s = []
ssBands = []
divide_to_ss = []
add_to_ss = []
allBands = []
divide_to_all = []
add_to_all = []
LSbandsIndex = []
CNNbandsIndex = []
elevBandIndex = 0

# This function sets scaling variables for different bands
def setBandScaling(allBands, neighborhood):
    global divide_to, add_to, add_to_s, divide_to_s, add_to_ss, divide_to_ss, divide_to_all, add_to_all, \
        LSbandsIndex, CNNbandsIndex,elevBandIndex

    LSbandsIndex = [bands.index(x) for x in LSbands]
    CNNbandsIndex = [bands.index(x) for x in CNNbands]
    elevBandIndex = sBands.index('elevation')
    divide_to_all = np.ones((len(allBands),neighborhood,neighborhood)).astype(np.float32)
    add_to_all = np.zeros((len(allBands),neighborhood,neighborhood)).astype(np.float32)
    for select in allBands:
        i = allBands.index(select)
        if select == 'DOY':
            divide_to_all[i,] = DOY_max
        elif select == 'sensor':
            divide_to_all[i,] = sensor_max
        elif select == 'ECO_ID':
            divide_to_all[i,] = ECOID_max
        elif select == 'elevation':
            divide_to_all[i,] = elevation_max - elevation_min
            add_to_all[i,] = -elevation_min
        elif select == 'slope':
            divide_to_all[i,] = slope_max
        elif select == 'aspect':
            divide_to_all[i,] = aspect_max
        elif select in ['NDVI','MSAVI2','SATVI','NLI','BSI','MNDWI','WNDWI','NDBI','ENDISI']:
            add_to_all[i,] = 1
        elif select == 'DD':
            divide_to_all[i,] = DD_max
        elif 'diss' in select:
            divide_to_all[i,] = GLCM_diss_max
        elif 'ent' in select:
            divide_to_all[i,] = GLCM_ent_max
        elif 'savg' in select:
            divide_to_all[i,] = GLCM_svag_max
        elif 'var' in select:
            divide_to_all[i,] = GLCM_var_max
        elif 'temp' in select:
            divide_to_all[i,] = temp_max - temp_min
            add_to_all[i,] = -temp_min
        elif 'Nrain' in select:
            divide_to_all[i,] = Nrain_max
        elif 'rain' in select:
            divide_to_all[i,] = rain_max
        divide_to = np.squeeze(divide_to_all[:len(bands)])
        add_to = np.squeeze(add_to_all[:len(bands)])
        divide_to_ss = divide_to_all[len(bands):len(bands)+len(ssBands),0,0]
        add_to_ss = add_to_all[len(bands):len(bands)+len(ssBands),0,0]
        if (neighborhood > 1) and addElevation2CNN:
            divide_to_s = divide_to_all[len(bands) + len(ssBands):, :, :]
            add_to_s = add_to_all[len(bands) + len(ssBands):, :, :]
        else:
            divide_to_s = divide_to_all[len(bands)+len(ssBands):,0,0]
            add_to_s = add_to_all[len(bands)+len(ssBands):,0,0]

def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx

def maskLandsatSR(image_array):
    # Bits 3, 5, 7, and 9  in Landsat7/8 show non-clear conditions
    # Get the pixel QA bands
    qa_index = bands.index('pixel_qa')
    radsat_index = bands.index('radsat_qa')
    for i in range(image_array.shape[0]):
        qa = image_array[i,qa_index,:,:].astype(np.int)
        radsat = image_array[i,radsat_index,:,:].astype(np.int)
        # All flags should be set to zero, indicating clear conditions.
        mask = ((qa & 0b1010101000) == 0) & (radsat == 0)
        image_array[i] = image_array[i]*mask
    image_array = np.delete(image_array, [qa_index, radsat_index], axis=1)
    return image_array

def toBinaryClass(target, class0):
    index0 = np.where(target == class0)
    index1 = np.where((target != class0) & (target > 0))      # 0 is invalid label
    target[index0] = 1      # 0 shows invalid label, so I used 1 instead of 0
    target[index1] = 2      # because 1 was used for class 0, 2 was used for class 1
    return target

def readData(blockName):
    global bands, ssBands, sBands, allBands

    # Read data structures from input numpy file
    with np.load(inFolder+blockName) as data:
        scenesList = data['scenes']
        bands = data['bands'].tolist()
        ss_image_years = data['annualDataYears'].tolist()
        ssBands = data['annualBands'].tolist()
        sBands = data['staticBands'].tolist()
        image_array = data['data'].astype(np.float32)
        ss_image = data['annualData'].astype(np.float32)
        s_image = data['staticData'].astype(np.float32)
        base_labels = data['labels'].reshape(-1)

    cl, cnt = np.unique(base_labels, return_counts=True)
    print('Base classes (including invalid 0 class) are {} with class counts equal to {}'.format(cl, cnt))
    if type(bands[0]) is bytes:
        bands = [x.decode('UTF-8') for x in bands]    # if read is in bytes, need to decode to string
    if ssBands:
        if type(ssBands[0]) is bytes:
            ssBands = [x.decode('UTF-8') for x in ssBands]
    if sBands:
        if type(sBands[0]) is bytes:
            sBands = [x.decode('UTF-8') for x in sBands]
    allBands = bands + ssBands + sBands

    if maskCloud:
        image_array = maskLandsatSR(image_array)
        bands.remove('pixel_qa')
        bands.remove('radsat_qa')
        allBands = bands + ssBands + sBands
    setBandScaling(allBands, neighborhood)

    if binaryClass0 != None:
        base_labels = toBinaryClass(base_labels, binaryClass0)
        cl, cnt = np.unique(base_labels, return_counts=True)
        print('Binary classes (including invalid 0 class) are {} with class counts equal to {}'.format(cl, cnt))

    # Read, sort, and filters list of scenes based on selected years and split the list to yearly portions
    sortIndex = np.lexsort((scenesList[:, 1], scenesList[:, 0]))
    scenesSorted = scenesList[sortIndex]
    delete_index = np.where((scenesSorted[:,0] < yearRange[0]) | (scenesSorted[:,0] > yearRange[1]))
    sortIndex = np.delete(sortIndex, delete_index, axis=0)
    scenesSorted = np.delete(scenesSorted, delete_index, axis=0)
    num_scenes = len(scenesSorted)
    yearsObserved = np.unique(scenesSorted[:,0]).tolist()
    yearsSplit = np.split(range(num_scenes), np.unique(scenesSorted[:, 0], return_index=True)[1][1:])
    numYears = len(yearsObserved)

    # Restrict years for training and testing
    if years2pickRule == 'fixed':
        pickedYearsIndex_train = [k for k in range(len(yearsObserved)) if yearsObserved[k] in years2pick_train]
        pickedYearsIndex_test = [k for k in range(len(yearsObserved)) if yearsObserved[k] in years2pick_test]
    elif years2pickRule == 'random':
        pickedYearsIndex_train = np.sort(np.random.choice(len(yearsObserved), years2pick_train, replace=False))
        pickedYearsIndex_test = np.sort(np.random.choice(len(yearsObserved), years2pick_test, replace=False))
    elif years2pickRule == 'systematic':
        pickedYearsIndex_train = np.arange(int(numYears / (2.0 * years2pick_train)), numYears, step=int(numYears / (1.0 * years2pick_train)))
        pickedYearsIndex_test = np.arange(int(numYears / (2.0 * years2pick_test)), numYears, step=int(numYears / (1.0 * years2pick_test)))
    else:  # rule = all
        pickedYearsIndex_train = range(len(yearsObserved))
        pickedYearsIndex_test = range(len(yearsObserved))
    yearsObserved_train = [yearsObserved[k] for k in pickedYearsIndex_train]
    yearsSplit_train = [yearsSplit[k] for k in pickedYearsIndex_train]
    yearsObserved_test = [yearsObserved[k] for k in pickedYearsIndex_test]
    yearsSplit_test = [yearsSplit[k] for k in pickedYearsIndex_test]

    return sortIndex, yearsObserved_train, yearsSplit_train, yearsObserved_test, yearsSplit_test, \
           image_array, ss_image_years, ss_image, s_image, base_labels

# This function is used when sampling scheme is 'fixed'
def sample_fixed(data, target, test_size, train_size=None):
    # class 0 is not included in the calculations in this function
    classes, counts = np.unique(target, return_counts=True)
    if 0 in classes:
        classes = classes[1:]
        counts = counts[1:]
    totalPoints = np.count_nonzero(target)
    if train_size == None:
        train_size = totalPoints - test_size
        trainFreqs = counts - testClassFreqs
        testFreqs = testClassFreqs
    else:
        trainFreqs = trainClassFreqs
        testFreqs = valClassFreqs
    if train_size + test_size > len(target):
        print('Error: train_size + test_size > target_size')
        return [],[],[],[]
    ixs = []
    for i in range(len(classes)):
        ixs.append(np.random.choice(np.nonzero(target==classes[i])[0], trainFreqs[i]+testFreqs[i], replace=False))
    # take same num of samples from all classes
    ix_train = np.concatenate([ixs[i][:trainFreqs[i]] for i in range(len(classes))])
    ix_test = np.concatenate([ixs[i][trainFreqs[i]:trainFreqs[i]+testFreqs[i]] for i in range(len(classes))])

    X_train = data[ix_train]
    X_test = data[ix_test]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test

# This function is used when sampling scheme is 'equal'
def sample_balanced(data, target, test_size, train_size=None):
    # test and train sizes are per class and can be ratios (<1) or exact numbers (integer)

    classes, counts = np.unique(target, return_counts=True)
    total_samples_per_class = min(counts)
    if train_size == None:
        if test_size < 1:
            train_size = 1-test_size
        else:
            train_size = total_samples_per_class - test_size
    if (test_size<=0) or (train_size<=0):
        return [],[],[],[]
    elif test_size<1:
        n_test_per_class = int(np.floor(total_samples_per_class*test_size))
        n_train_per_class = int(np.floor(total_samples_per_class*train_size))
        if (n_test_per_class == 0) or (n_train_per_class == 0):
            return [], [], [], []
    elif (test_size+train_size <= total_samples_per_class):
        n_test_per_class = int(np.floor(test_size))
        n_train_per_class = int(np.floor(train_size))
    else:
        return [],[],[],[]

    ixs = []
    for cl in classes:
        ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class, replace=False))
    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])

    X_train = data[ix_train]
    X_test = data[ix_test]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test

# Main function for sampling test points set
def sampleLabels_test(image_array, base_labels, test_size):

    imageHeight = image_array.shape[2]
    imageWidth = image_array.shape[3]
    counts = np.bincount(base_labels.astype(int))
    total = sum(counts)
    invalids = counts[0]
    valids = total - invalids
    a1 = invalids * (float(test_size) / valids)
    index = np.arange(len(base_labels))
    if len(base_labels) > int(a1+test_size):
        if samplingScheme == 'equal':
            ind_trval, ind_test, Y_trval, Y_test = sample_balanced(index, base_labels, test_size=test_size)
        elif samplingScheme == 'stratified':
            ind_trval, ind_test, Y_trval, Y_test = train_test_split(index, base_labels, test_size=int(a1+test_size),
                                                                    stratify=base_labels)
        elif samplingScheme == 'fixed':
            ind_trval, ind_test, Y_trval, Y_test = sample_fixed(index, base_labels, test_size=test_size)
        else:
            ind_trval, ind_test, Y_trval, Y_test = train_test_split(index, base_labels, test_size=int(a1 + test_size))
        if ind_trval == []:
            print('Not enough data to extract {} test points.'.format(test_size))
            return  [], [], []
    else:
        print('Not enough data to extract {} test points.'.format(test_size))
        return  [], [], []
    valids_Y_trval = Y_trval!=0
    Y_trval = Y_trval[valids_Y_trval]
    ind_trval = ind_trval[valids_Y_trval]
    valids_Y_test = Y_test!=0
    Y_test = Y_test[valids_Y_test]
    ind_test = ind_test[valids_Y_test]
    test_points = np.zeros((imageHeight, imageWidth))
    row_col = [divmod(i, imageWidth) for i in ind_test]
    for i in range(len(row_col)):
        test_points[row_col[i][0], row_col[i][1]] = Y_test[i]
    return test_points, ind_trval, Y_trval

# Main function for sampling training/validation points set
def sampleLabels_train(image_array, ind_trval, Y_trval, train_val_sizes):

    imageHeight = image_array.shape[2]
    imageWidth = image_array.shape[3]
    train_points = np.zeros((imageHeight, imageWidth))
    val_points = np.zeros((imageHeight, imageWidth))
    train_size = train_val_sizes[0]
    val_size = train_val_sizes[1]
    if train_size == 0:
        train_size = len(ind_trval) - val_size
    if (len(ind_trval) >= train_size+val_size) or (samplingScheme == 'fixed'):
        if samplingScheme == 'equal':
            ind_train, ind_val, Y_train, Y_val = sample_balanced(ind_trval, Y_trval, train_size=train_size,
                                                                 test_size=val_size)
        elif samplingScheme == 'stratified':
            ind_train, ind_val, Y_train, Y_val = train_test_split(ind_trval, Y_trval, train_size=train_size,
                                                                  test_size=val_size, stratify=Y_trval)
        elif samplingScheme == 'fixed':
            ind_train, ind_val, Y_train, Y_val = sample_fixed(ind_trval, Y_trval, train_size=train_size,
                                                                  test_size=val_size)
        else:
            ind_train, ind_val, Y_train, Y_val = train_test_split(ind_trval, Y_trval, train_size=train_size,
                                                                  test_size=val_size)
        if ind_train == []:
            print('Not enough data to extract {} train and {} validation points.'.format(train_size, val_size))
            return [], []
    else:
        print('Not enough data to extract {} train and {} validation points.'.format(train_size, val_size))
        return [], []
    row_col = [divmod(i, imageWidth) for i in ind_train]
    for i in range(len(row_col)):
        train_points[row_col[i][0], row_col[i][1]] = Y_train[i]
    row_col = [divmod(i, imageWidth) for i in ind_val]
    for i in range(len(row_col)):
        val_points[row_col[i][0], row_col[i][1]] = Y_val[i]
    return train_points, val_points

# Main function for extracting data for given set of selected points
def createOutput(outputFile, neighborhood, sortIndex, yearsObserved, yearsSplit, image_array,
                 ss_image_year, ss_image, s_image, selected_points):

    imageHeight = image_array.shape[2]
    imageWidth = image_array.shape[3]
    num_years = len(yearsObserved)
    yearlyDataPatches = []
    annualDataVector = []
    staticDataVector = []
    spatialDataVector = []
    outputLabels = []
    invalidBlobs = 0
    max_seq_len = 0
    nb = int(neighborhood / 2)
    rows, cols = np.where(selected_points > 0)
    valids = zip(rows, cols)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(outputFile, options=options)
    data_len = 0
    print('Processing sampled data to generate output file {}...'.format(outputFile))
    for row, col in valids:
        if (row < nb) or (row > imageHeight - nb - 1) or (col < nb) or (col > imageWidth - nb - 1):
            continue
        for i in range(num_years):
            thisYearIndex = sortIndex[yearsSplit[i]]
            thisYearBlob = image_array[thisYearIndex, :, row - nb:row + nb + 1, col - nb:col + nb + 1]
            # Check the central pixel's Landsat band values for each timestamp
            if neighborhood == 1:
                outputBlob = np.array([x for x in np.squeeze(thisYearBlob) if x[LSbandsIndex].any()])
            else:
                outputBlob = np.array([x for x in np.squeeze(thisYearBlob) if x[LSbandsIndex,nb,nb].any()])
            if len(outputBlob) > 0:
                if remove_dups:
                    # remove duplicate DOY entries
                    if neighborhood == 1:
                        _, index = np.unique(outputBlob[:, 0], return_index=True)
                    else:
                        _, index = np.unique(outputBlob[:, 0, nb, nb], return_index=True)
                    outputBlob = outputBlob[index,:]
                # reduce to selected DOYs
                if reduceTime:
                    if neighborhood == 1:
                        index = np.unique([closest(outputBlob[:, 0] , l) for l in seasonDOYs])
                    else:
                        index = np.unique([closest(outputBlob[:, 0, nb, nb], l) for l in seasonDOYs])
                    outputBlob = outputBlob[index, :]
                ss_output = ss_image[ss_image_year.index(yearsObserved[i]), :, row, col]
                if (neighborhood > 1) and addElevation2CNN:
                    s_output = s_image[:,row - nb:row + nb + 1, col - nb:col + nb + 1]
                else:
                    s_output = s_image[:, row, col]
                newDataPatch = ((outputBlob + add_to) / divide_to).astype(np.float32)
                newAnnualData = ((ss_output + add_to_ss) / divide_to_ss).astype(np.float32)
                newStaticData = ((s_output + add_to_s) / divide_to_s).astype(np.float32)
                if neighborhood > 1:
                    newSpatialPatch = newDataPatch[:,CNNbandsIndex,:,:]
                    if addElevation2CNN:
                        elevSpatialData = newStaticData[elevBandIndex, :, :]
                        tempArray = np.repeat(elevSpatialData[np.newaxis, np.newaxis, :, :], newDataPatch.shape[0], axis=0)
                        newSpatialPatch = np.concatenate((newSpatialPatch, tempArray), axis=1)
                        newStaticData = newStaticData[:, nb, nb]
                    newDataPatch = newDataPatch[:,:,nb,nb]
                newLabel = selected_points[row, col].astype(np.int64)
                yearlyDataPatches.append(newDataPatch)
                annualDataVector.append(newAnnualData)
                staticDataVector.append(newStaticData)
                if neighborhood > 1:
                    spatialDataVector.append(newSpatialPatch)
                outputLabels.append(newLabel)
                if max_seq_len < outputBlob.shape[0]:
                    max_seq_len = max(max_seq_len, outputBlob.shape[0])
            else:
                invalidBlobs += 1

    data_len = len(yearlyDataPatches)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    np.random.shuffle(indices)
    print('-> writing {} records ...'.format(data_len))
    for i in indices:
        if neighborhood == 1:
            features = tf.train.Features(feature={
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[yearlyDataPatches[i].tostring()])),
                'annualData': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annualDataVector[i].tostring()])),
                'staticData': tf.train.Feature(bytes_list=tf.train.BytesList(value=[staticDataVector[i].tostring()])),
                'rows': tf.train.Feature(int64_list=tf.train.Int64List(value=[yearlyDataPatches[i].shape[0]])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[outputLabels[i]]))
            })
        else:
            features = tf.train.Features(feature={
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[yearlyDataPatches[i].tostring()])),
                'annualData': tf.train.Feature(bytes_list=tf.train.BytesList(value=[annualDataVector[i].tostring()])),
                'staticData': tf.train.Feature(bytes_list=tf.train.BytesList(value=[staticDataVector[i].tostring()])),
                'spatialData': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spatialDataVector[i].tostring()])),
                'rows': tf.train.Feature(int64_list=tf.train.Int64List(value=[yearlyDataPatches[i].shape[0]])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[outputLabels[i]]))
            })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()
    return data_len, max_seq_len, invalidBlobs

# Main script
for block in files:
    if 'points' in block:
        continue
    print('\nProcessing input file {}:'.format(block))
    start = time.time()
    sortIndex, yearsObserved_train, yearsSplit_train, yearsObserved_test, yearsSplit_test, \
    image_array, ss_image_year, ss_image, s_image, base_labels = readData(block)
    if len(yearsObserved_train)*len(yearsObserved_test) == 0:
       print('No data available for picked train or test years.')
       continue
    num_scenes = len(sortIndex)
    print('Working on neighborhood size of {}.'.format(neighborhood))
    if reduceTime:
        outputFile = outFolder + block[:11] + '_seasonal_K' + str(neighborhood) + '_'
    else:
        outputFile = outFolder + block[:11] + '_K' + str(neighborhood) + '_'
    if binaryClass0 != None:
        outputFile = outputFile + 'B'
    if samplingScheme == 'equal':
        outputFile = outputFile + 'EQ'
    elif samplingScheme == 'stratified':
        outputFile = outputFile + 'SS'
    elif samplingScheme == 'fixed':
        outputFile = outputFile + 'FX'

    # Generate/reuse test points
    outputFile1 = outputFile + '_(' + str(data_sizes[2]) + ')'
    if reuseTestPoints:
        try:
            with np.load(outputFile1+'_test_points.npz') as data:
                test_points = data['test_points']
                ind_trval = data['ind_trval']
                Y_trval = data['Y_trval']
        except:
            print('{} test points file not found.'.format(outputFile1))
            continue
    else:
        test_points, ind_trval, Y_trval = sampleLabels_test(image_array, base_labels, data_sizes[2])
        if len(test_points) == 0:
            continue
        np.savez_compressed(outputFile1+'_test_points.npz', test_points=test_points, ind_trval=ind_trval, Y_trval=Y_trval)

    # Generate test dataset
    test_len, max_seq_len3, invBlobs3 = createOutput(outputFile1+'_test.gz', neighborhood, sortIndex,
        yearsObserved_test, yearsSplit_test, image_array, ss_image_year, ss_image, s_image, test_points)

    for setNumber in repeat:
        if data_sizes[0] != 0:
            outputFile2 = outputFile + '_' + str(data_sizes[:2]).replace(' ','') + '_' + str(setNumber)
        else:
            outputFile2 = outputFile + '_(rest,' + str(data_sizes[1]) + ')'+ '_'+str(setNumber)
        # Generate train/val points
        if reuseTrainPoints:
            try:
                with np.load(outputFile2 + '_train_points.npz') as data:
                    train_points = data['train_points']
                    val_points = data['val_points']
            except:
                print('Train points file#{} not found.'.format(setNumber))
                continue
        else:
            train_points, val_points = sampleLabels_train(image_array, ind_trval, Y_trval, data_sizes[:2])
            if len(train_points) == 0:
                continue
            np.savez_compressed(outputFile2+'_train_points.npz',train_points=train_points,val_points=val_points)

        # Generate train/val datasets
        train_len, max_seq_len1, invBlobs1 = createOutput(
            outputFile2+'_train.gz', neighborhood, sortIndex, yearsObserved_train, yearsSplit_train,
            image_array, ss_image_year, ss_image, s_image, train_points)
        val_len, max_seq_len2, invBlobs2  = createOutput(
            outputFile2+'_val.gz', neighborhood, sortIndex, yearsObserved_test, yearsSplit_test,
            image_array, ss_image_year, ss_image, s_image, val_points)
        with open(outputFile2+'_config.txt', 'w') as f:
            f.write('Total number of Landsat scenes read = {:,}, remove duplicates = {}, Cloud masking = {}\n'.
                    format(num_scenes, remove_dups, maskCloud))
            f.write('Observed years for training/validation = {}\n'.format(yearsObserved_train))
            f.write('Observed years for testing = {}\n'.format(yearsObserved_test))
            f.write('Training data spatial neighborhood size = {}\n'.format(neighborhood))
            if neighborhood > 1:
                if addElevation2CNN:
                    f.write('Spatial data saved for bands: {}\n'.format(CNNbands + ['elevation']))
                else:
                    f.write('Spatial data saved for bands: {}\n'.format(CNNbands))
            if reuseTrainPoints:
                f.write('Reusing training/validation points data from file {}\n'.format(outputFile2))
            if reuseTestPoints:
                f.write('Reusing test points data from file {}\n'.format(outputFile1))
            if binaryClass0 != None:
                f.write('Classification is binary with binary class0 = {}\n'.format(binaryClass0))
            allPoints = [train_points, val_points, test_points]
            title = ['train', 'val','test']
            for i in range(len(allPoints)):
                points = allPoints[i].reshape(-1)
                valid_points = points[points != 0]
                cl, cnt = np.unique(valid_points, return_counts=True)
                f.write('- {} classes = {}, counts = {}, total = {}\n'.format(title[i], cl, cnt,sum(cnt)))
            f.write(
                'Total number of used yearly sequences = {:,} (invalid sequences = {:,}) \n'.
                    format(train_len+val_len+test_len, invBlobs1+invBlobs2+invBlobs3))
            f.write('Number of training sequences = {:,}\n'.format(train_len))
            f.write('Number of validation sequences = {:,}\n'.format(val_len))
            f.write('Number of testing sequences = {:,}\n'.format(test_len))
            f.write('Maximum number of records in the final sequences = {}\n'.format(
                max(max_seq_len1, max_seq_len2, max_seq_len3)))
            f.write('Number of features per record (dynamic, annual, and static) = {}, {}, {}\n'
                    .format(len(bands), len(ssBands), len(sBands)))
            if reduceTime:
                f.write('Seasonal DOYs = {}\n'.format(seasonDOYs))
            f.write('Included bands: {}\n'.format(allBands))
            f.write('\nFile used to generate dataset:{}\n'.format(block))
    del image_array, ss_image_year, ss_image, s_image, base_labels
    gc.collect()
    print('\nTotal run time for block {} was {:.0f} seconds.\n'.format(block, time.time() - start))


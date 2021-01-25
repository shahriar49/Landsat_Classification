# Update in V5: update the included fields and remove all non-LSTM cases
# Shahriar S. Heydari, 12/4/2020

import numpy as np
import os, fnmatch, sys

baseFolder = 'Z:\\WIP\\eregpu1\\texts\\'
pid_prefix = 'allblocks2_plain_full_12days_'
CNN_flag = False
append2File = False
test_batch_size = 1024
baseBandNames = ['DOY', 'sensor', 'blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2', 'elevation', 'aspect', 'slope',
                 'ENDISI', 'DD_ent_5x64', 'ENDISI_ent_15x64', 'blue_savg_15x64']
outFolder = ''
files = sorted(os.listdir(baseFolder))
text_files = files = fnmatch.filter(files, pid_prefix+'*.txt')
outFile = outFolder + 'runResults.csv'

if append2File:
    outList = open(outFile, 'a')
else:
    outList = open(outFile, 'w')
    if CNN_flag:
        outList.write('run,PID,FID,features,CNN_config,LSTM_classifier,dropout,params,optimizer,optimizer_params,'
                      'recurrent_reg,cnn_reg,dense_act,cnn_act,batch_size,train_size,val_size,test_size,'
                      'runtime,epochs,train_acc,val_acc,test_acc,avg_F1,min_F1,min_class,'
                      'F1_water,F1_imp,F1_grass,F1_forest,F1_bare,F1_agri,F1_wetland\n')
    else:
        outList.write('run,PID,FID,features,LSTM_classifier,dropout,params,optimizer,optimizer_params,'
                      'recurrent_reg,dense_act,batch_size,train_size,val_size,test_size,'
                      'runtime,epochs,train_acc,val_acc,test_acc,avg_F1,min_F1,min_class,'
                      'F1_water,F1_imp,F1_grass,F1_forest,F1_bare,F1_agri,F1_wetland\n')

for file in text_files:
    print('Reading file',file,'...', end="")
    p1 = file.find('[')
    pid = file[:p1]
    ff = file[len(pid)+1:]
    # Extract run# (repetition#), which is either a number enclosed in square brackets in the pid_prefix or its last character
    p2 = ff.find(']')
    run_number = ff[:p2]
    # Extract feature id, which is between two underscore characters after [run_number]
    temp = ff[p2+2:]    # skip the first underscore
    fid = temp[:temp.find('_')]
    if fid == 'ATE':
        fid = '1000'
    if fid == 'APR':
        fid = '1001'
    if fid == 'ECO':
        fid = '1100'
    if fid == 'NTM':
        fid = '1200'
    if fid == 'NTS':
        fid = '1201'
    if fid == 'NPM':
        fid = '1202'
    if fid == 'NPS':
        fid = '1203'

    # Extract network configuration from filename
    conf = ''
    if CNN_flag:
        p1 = file.find('CNN')
        p2 = file[p1+1:].find('_')
        if p2 == -1:
            conf = conf + file[p1:-4]
        else:
            conf = conf + file[p1:p1+p2+1]
    else:# 'LSTM' in file:
        p1 = file.find('LSTM')
        p2 = file[p1+1:].find('_')
        if p2 == -1:
            conf = conf + file[p1:-4]
        else:
            conf = conf + file[p1:p1+p2+1]
    # Extract CNN configuration from file name
    if CNN_flag:
        p1 = conf.find('LSTM')
        CNNconf = conf[:p1]
        conf = conf[p1:]
    else:
        CNNconf = ''
    # Extract Dropout from file name
    p1 = conf.find('Drop')
    if p1 != -1:
        dropout = 'Dropout(' + conf[p1+5:-1].replace(',','-') + ')'
        LSTMconf = conf[:p1]
    else:
        LSTMconf = conf
        dropout = ''

    # Read whole file contents
    with open(baseFolder+file, 'r') as f:
        contents = f.read()
    try:
        # Extract number of parameters
        p1 = contents.find('Total params:')
        if p1 != -1:
            p2 = contents[p1+13:].find('\n')
            params = int(contents[p1+13:p1+13+p2].replace(',','').strip())
        else:
            params = ''

        # Extract Optimizer type
        p1 = contents.find('Optimizer')
        # Check if optimizer parameters are given or not
        temp = contents[p1+10:]
        p2 = temp.find('{')
        if p2 == -1:
            p2 = temp.find('\n')
            optimizer = temp[:p2].strip()
            optimizer_params = ''
        else:
            optimizer = temp[:p2].strip()
            optimizer_params = temp[p2+1:temp.find('}')].replace(',','-')

        # Extract recurrent and CNN regularizers
        p1 = contents.find('recurrent_regularizer:')
        if p2 == -1:
            rec_reg = ''
        else:
            temp = contents[p1:]
            rec_reg = temp[22:temp.find('\n')].strip()
        p1 = contents.find('CNN_regularizer:')
        if p2 == -1:
            cnn_reg = ''
        else:
            temp = contents[p1:]
            cnn_reg = temp[17:temp.find('\n')].strip()

        # Extract dense activation
        p1 = contents.find('Dense layer activation:')
        if p2 == -1:
            dense_act = ''
        else:
            temp = contents[p1:]
            dense_act = temp[23:temp.find('\n')].strip()
        p1 = contents.find('CNN layer activation:')
        if p2 == -1:
            cnn_act = ''
        else:
            temp = contents[p1:]
            cnn_act = temp[22:temp.find('\n')].strip()


        # Extract Selected bands
        p1 = contents.find('Selected recurrent features:')
        p2 = contents[p1+29:].find(']')
        bandNames = eval(contents[p1+29:p1+29+p2+1].strip())
        p1 = contents.find('Selected static features:')
        p2 = contents[p1+26:].find(']')
        bandNames = bandNames + eval(contents[p1+26:p1+26+p2+1].strip())
        p1 = contents.find('Selected annual features:')
        p2 = contents[p1+26:].find(']')
        bandNames = bandNames + eval(contents[p1+26:p1+26+p2+1].strip())
        additionalBands = list(set(bandNames) - set(baseBandNames))

        # Find input data size
        p1 = contents.find('Training, validation, and testing steps =')
        temp = contents[p1+41:]
        p2 = temp.find(')')
        steps = eval(temp[:p2+1].strip())
        train_steps = steps[0]
        val_steps = steps[1]
        test_steps = steps[2]

        # Find training batch size
        p1 = contents.find('Training mini-batch size:')
        temp = contents[p1+25:]
        p2 = temp.find('\n')
        batch_size = int(temp[:p2])
        train_size = train_steps*batch_size
        val_size = val_steps*batch_size
        test_size = test_steps*test_batch_size

        # Find model run time
        p1 = contents.find('Model training time was ')
        temp = contents[p1+24:]
        p2 = temp.find(' ')
        run_time = int(temp[:p2].strip())

        # Find number of epochs (including early stopping case)
        p1 = contents.find('Max epochs: ')
        temp = contents[p1+11:]
        p2 = temp.find('\n')
        temp = temp[:p2]
        p3 = temp.find(',')
        if p3 == -1:
            epochs = int(temp.strip())
        else:
            p2 = temp.find('epoch ')
            epochs = int(temp[p2+6:].strip())

        # Find the line with train and validation accuracies
        p1 = contents.find('Best value of train accuracy was')
        temp = contents[p1+32:]
        p2 = temp.find('and')
        train_acc = float(temp[:p2-1].strip())
        p1 = contents.find('Best value of validation accuracy was')
        temp = contents[p1+37:]
        p2 = temp.find('and')
        val_acc = float(temp[:p2-1].strip())

        # Find the classification results table and build it
        temp = contents[contents.find('support')+9:]
        line_end = temp.find('\n')
        table = np.fromstring(temp[:line_end].strip(), dtype=float, sep=' ')
        temp = temp[line_end+1:]
        line_end = temp.find('\n')
        while temp[:line_end].strip() != '':
            table = np.vstack((table, np.fromstring(temp[:line_end].strip(), dtype=float, sep=' ')))
            temp = temp[line_end+1:]
            line_end = temp.find('\n')
        # Read the line containing test accuracy (after the classification results table)
        p1 = temp.find('accuracy')
        if p1 == -1:
            p1 = temp.find('micro avg')
        temp = temp[p1:]
        p2 = temp.find('\n')
        line = temp[p1+8:p2].strip()
        temp = temp[p2+1:].strip()
        p1 = line.find(' ')
        test_acc = float(line[:p1].strip())
        # Read the line after the above line and pass first 4 words to reach average F1 value
        p1 = temp.find(' ')
        temp = temp[p1+1:].strip()
        p1 = temp.find(' ')
        temp = temp[p1+1:].strip()
        p1 = temp.find(' ')
        temp = temp[p1+1:].strip()
        p1 = temp.find(' ')
        temp = temp[p1+1:].strip()
        p1 = temp.find(' ')
        average_F1 = float(temp[:p1].strip())
        # Look at F1 values to find the minimum F1 and worst class
        F1s = table[:,3]
        #supports = table[:,4]
        #average_F1 = sum(F1s)/np.count_nonzero(supports)
        min_F1_class = F1s.argmin()
        min_F1_value = F1s[min_F1_class]

        # Write results to the output csv file
        outList.write('{},{},{},'.format(run_number,pid.replace(',','.'),fid))
        if additionalBands != []:
            additionalBands.sort()
            outList.write('&'.join('{:}'.format(f1) for f1 in additionalBands))
            outList.write(',')
        else:
            outList.write(',')

        if CNN_flag:
            outList.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'.format(CNNconf.replace(',', '.'),LSTMconf.replace(',', '.'),
                        dropout, params, optimizer, optimizer_params, rec_reg, cnn_reg, dense_act, cnn_act,
                        batch_size, train_size, val_size, test_size, run_time, epochs))
        else:
            outList.write('{},{},{},{},{},{},{},{},{},{},{},{},{},'.format(LSTMconf.replace(',','.'),
                        dropout, params, optimizer, optimizer_params, rec_reg, dense_act,
                        batch_size, train_size,val_size,test_size,run_time,epochs))

        outList.write('{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},'.
                      format(train_acc,val_acc,test_acc,average_F1,min_F1_value,min_F1_class))
        outList.write(','.join('{:.5f}'.format(f1) for f1 in F1s))
        outList.write('\n')
        print(' done')
    except:
        print(' failed')
        pass

outList.close()


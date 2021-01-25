# Update in V5: update the included fields and remove all non-LSTM cases
# Shahriar S. Heydari, 12/4/2020

import numpy as np
import os, fnmatch, sys

baseFolder = 'D:\\Shahriar\\OneDrive - SUNY ESF\\Thesis\\Results2\\6.RevisitBase\\RF\\test_pruning_2M\\'
pid_prefix = ''
append2File = False
addSupportPerClass = True
outFolder = baseFolder
files = sorted(os.listdir(baseFolder))
text_files = files = fnmatch.filter(files, pid_prefix+'*.txt')
outFile = outFolder + 'runResults.csv'

if append2File:
    outList = open(outFile, 'a')
else:
    outList = open(outFile, 'w')

if addSupportPerClass:
    outList.write('file,Num_params,runtime,train_acc,val_acc,test_acc,avg_F1,min_F1,min_class,'
                  'Water_sup,Water_F1,Imp_sup,Imp_F1,Grass_sup,Grass_F1,Forest_sup,Forest_F1,'
                  'Bare_sup,Bare_F1,Agri_sup,Agri_F1,Wetlan_sup,Wetlan_F1\n')
else:
    outList.write('file,Num_params,runtime,train_acc,val_acc,test_acc,avg_F1,min_F1,min_class,'
                  'F1_water,F1_imp,F1_grass,F1_forest,F1_bare,F1_agri,F1_wetland\n')

for file in text_files:
    print('Reading file',file,'...', end="")

    # Read whole file contents
    with open(baseFolder+file, 'r') as f:
        contents = f.read()
    try:
        # Extract number of parameters
        p1 = contents.find('raining time')
        if p1 != -1:
            temp = contents[p1+12:]
            p2 = temp.find('\n')
            temp = temp[:p2].replace(':','').replace('=','').replace('was','').replace('seconds','').strip()
            run_time = int(temp)
        else:
            run_time = 0

        # Extract number of parameters
        p1 = contents.find('Total params:')
        if p1 != -1:
            p2 = contents[p1+13:].find('\n')
            params = int(contents[p1+13:p1+13+p2].replace(',','').strip())
        else:
            p1 = contents.find('Total number of forest nodes:')
            if p1 != -1:
                p2 = contents[p1 + 30:].find(',')
                if p2 == -1:
                    p2 = contents[p1 + 30:].find('\n')
                params = int(contents[p1 + 30:p1 + 30 + p2].replace(',', '').strip())
            else:
                params = ''

        # Find the line with train and validation accuracies
        p1 = contents.find('Best value of train accuracy was')
        if p1 != -1:
            temp = contents[p1+32:]
            p2 = temp.find('and')
            train_acc = float(temp[:p2-1].strip())
            p1 = contents.find('Best value of validation accuracy was')
            temp = contents[p1+37:]
            p2 = temp.find('and')
            val_acc = float(temp[:p2-1].strip())
        else:
            p1 = contents.find('train and validation accuracies was')
            if p1 != -1:
                temp = contents[p1+36:]
                p2 = temp.find('and')
                train_acc = float(temp[:p2-1].strip())
                temp = temp[p2+3:]
                p2 = temp.find('\n')
                val_acc = float(temp[:p2-1].strip())
            else:
                train_acc = 0
                val_acc = 0

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
        f1_list = list(table[:,3])
        support_list = list(table[:,4])
        f1_min_class = table[f1_list.index(min(f1_list)),0]
        f1_min = table[f1_list.index(min(f1_list)),3]
        f1_mean = sum(f1_list) / len(f1_list)
        classes = [0,1,2,3,4,5,6]
        missing = [x for x in classes if not x in list(table[:,0])]
        for c in range(len(missing)):
            f1_list = f1_list + [-1]
            support_list = support_list + [0]
        classes, f1_list, support_list = zip(*sorted(zip(list(table[:,0]) + missing, f1_list, support_list)))

        p1 = temp.find('accuracy')
        if p1 == -1:
            p1 = temp.find('micro avg')
        temp = temp[p1:]
        p2 = temp.find('\n')
        line = temp[p1+8:p2].strip()
        temp = temp[p2+1:].strip()
        p1 = line.find(' ')
        test_acc = float(line[:p1].strip())
        # Write results to the output csv file
        if addSupportPerClass:
            outList.write('{},{},{:d},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},'.
                          format(file.replace('.txt','').replace(',','.'),params,run_time,train_acc,val_acc,test_acc,
                                 f1_mean,f1_min,f1_min_class))
            for i in range(len(classes)):
                outList.write('{},{},'.format(support_list[i],f1_list[i]))
        else:
            outList.write('{},{},{:d},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{},'.
                          format(file.replace('.txt','').replace(',','.'),params,run_time,train_acc,val_acc,test_acc,
                                 f1_mean,f1_min,f1_min_class))
            outList.write(','.join('{:.5f}'.format(f1) for f1 in f1_list))
        outList.write('\n')
        print(' done')
    except:
        print(' failed')
        pass

outList.close()


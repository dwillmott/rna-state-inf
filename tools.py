import os
import numpy as np
import time
import sys


def makernndirs():    
    dirs = ['rnns', 'predictions', 'predictions/rnn']
    
    for d in ['rnns', 'predictions', 'predictions/rnn']:
        if not os.path.exists(d):
            os.makedirs(d)


def makehmmdirs():
    dirs = ['hmms', 'predictions', 'predictions/rnn']
    
    for d in ['hmms', 'predictions', 'predictions/rnn']:
        if not os.path.exists(d):
            os.makedirs(d)


def runmetrics(actual, predicted, setname, machine):
    # 
    # classes are 0 (unpaired), 1 (paired), 2 (end of sequence)
    
    testnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM',
                'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
                'syne', 'ecoli', 'subtilis', 'desulfuricans',
                'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    
    # predicted & actual from RNN is an np array of shape (samples, timesteps, classes): probability predictions
    if machine == 'RNN':
        sequencelengths = [np.trim_zeros(np.sum(act, axis = -1)).size for act in actual]
        y = [np.argmax(actual[i], axis = -1)[:sequencelengths[i]] for i in range(actual.shape[0])]
        yhat = [np.argmax(predicted[i], axis = -1)[:sequencelengths[i]] for i in range(predicted.shape[0])]
    
    if machine == 'HMM':
        y = actual
        yhat = predicted
    
    print('\n%s Metrics\n' % setname)
    
    if setname == 'Test Set':
        print("          NAME  | TOTAL  |  TRUE  FALSE  | TPOS  TNEG  FPOS  FNEG  |   TPOS    TNEG    FPOS    FNEG  |    ACC    PPV    SEN")
    
    metrics = []
    for i, (y_seq, yhat_seq) in enumerate(zip(y, yhat)):
        tp = np.count_nonzero(y_seq * yhat_seq)
        fp = np.count_nonzero((1-y_seq) * yhat_seq)
        tn = np.count_nonzero((1-y_seq) * (1 -yhat_seq))
        fn = np.count_nonzero(y_seq * (1-yhat_seq))
        
        metrics.append([tp, fp, tn, fn])
    
        if setname == 'Test Set':
            # print metrics for each test set sequence
            total = tp + fn + tn + fp
            acc = (tp + tn)/total
            ppv = tp/(tp + fp + 10e-5)
            sen = tp/(tp + fn + 10e-5)
            
            printstring = (testnames[i],total,tp+tn,fp+fn,tp,tn,fp,fn,tp/total,tn/total,fp/total,fn/total,acc,ppv,sen)
            
            print('%14s  |  %4d  |  %4d   %4d  | %4d  %4d  %4d  %4d  |  %.3f   %.3f   %.3f   %.3f  |  %.3f  %.3f  %.3f' % (printstring))
    
    tp, fp, tn, fn = np.sum(metrics, axis = 0)
    total = tp + fn + tn + fp
    
    print('\nOf %d nucleotides:\n' % (total))
    
    print('%d true positives (%.2f%%)' % (tp, (float(tp)/total)*100))
    print('%d true negatives (%.2f%%)' % (tn, (float(tn)/total)*100))
    print('%d false positives (%.2f%%)' % (fp, (float(fp)/total)*100))
    print('%d false negatives (%.2f%%)\n' % (fn, (float(fn)/total)*100))
    
    print('%d true (%.2f%%)' % (tp+tn, ((float(tp)+float(tn))/total)*100))
    print('%d false (%.2f%%)\n' % (fp+fn, ((float(fp)+float(fn))/total)*100))
    
    print('ppv: %.4f' % (tp/(tp + fp + 10e-5)))
    print('sen: %.4f\n' % (tp/(tp + fn + 10e-5)))
    
    return (float(tn + tp)/total, float(fp)/total, float(fn)/total)



def writeoutput(sequence, actual, predicted, machine, epoch = None, k = None):
    
    testnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM',
               'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
               'syne', 'ecoli', 'subtilis', 'desulfuricans',
               'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    
    ndict = {0 : 'A', 1 : 'C', 2 : 'G', 3 : 'U', 4 : 'X'}
    
    
    if machine == 'RNN':
        sequencelengths = [np.trim_zeros(np.sum(act, axis = -1)).size for act in actual]
        sequence = [np.argmax(sequence[i], axis = -1)[:sequencelengths[i]] for i in range(sequence.shape[0])]
        actual = [np.argmax(actual[i], axis = -1)[:sequencelengths[i]] for i in range(actual.shape[0])]
        probability = [predicted[i,:,1][:sequencelengths[i]] for i in range(predicted.shape[0])]
        predicted = [np.argmax(predicted[i], axis = -1)[:sequencelengths[i]] for i in range(predicted.shape[0])]
        
        outputdirectory = 'text/rnn/epoch%02d/' % (epoch,)
        headerstring = 'Epoch %02d' % (epoch,)
        
    if machine == 'HMM':
        outputdirectory = 'text/hmm/order%d/' % (k,)
        headerstring = 'Order %d' % (k,)
    
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
    
    # text output
    for j in range(16):
        
        textfile = open(outputdirectory + '%s-stateprediction.txt' % (testnames[j],), 'w+')
        textfile.write('%s  %s \n\n' % (headerstring, testnames[j]))
        
        if machine == "RNN":
            for i, (x, y, yhat, prob) in enumerate(zip(sequence[j], actual[j], predicted[j], probability[j])):
                printstr = '%d  %s  %d  %d  %0.4f\n' % (i+1, ndict[x], y, yhat, prob)
                textfile.write(printstr)
        if machine == "HMM":
            for i, (x, y, yhat) in enumerate(zip(sequence[j], actual[j], predicted[j])):
                printstr = '%d  %s  %d  %d\n' % (i+1, ndict[x], y, yhat)
                textfile.write(printstr)
        textfile.write('\n')
        textfile.close()

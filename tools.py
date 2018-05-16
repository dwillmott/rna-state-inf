import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


# print(machine & dataset info at start of training
def printstart(layers, trainsize, testsize, batchsize, timesteps, datadim, epochs, reg, dropprob, lr, lrdecay, optim):
    print('\nMACHINE:\n')
    print(str(len(layers)) + '-layer RNN')
    #print('Layer Sizes:', [datadim] + [l.output_dim for l in layers if 'dropout' not in l.name]
    print('Optimizer: %s' % (optim,))
    print('Learning rate: %.4f' % (lr,))
    print('Learning rate decay: %.2f' % (lrdecay,))
    if reg:
        print('L2 regularization coefficient = %.4f' % (reg,))
    else:
        print('No L2 regularization')
    if dropprob:
        print('Dropout with drop probability %.2f' % (dropprob,))
    else:
        print('No dropout')
    
    print('\nDATASET:\n')
    print('%d training sequences, %d test sequences' % (trainsize, testsize))
    print('%d training batches of size %d' % (trainsize/batchsize, batchsize))
    print('%d test batches of size %d' % (testsize/batchsize, batchsize))
    print('Using first %d time steps' % (timesteps,))
    print('%d epochs\n\n' % (epochs,))


def makernndirs():
    zsnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM', 'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum', 'syne', 'ecoli', 'subtilis', 'desulfuricans', 'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    
    dirs = ['rnns', 'text']
    dirs += ["text/%.5s" % (zsname) for zsname in zsnames]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def makehmmdirs():
    zsnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM', 'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum', 'syne', 'ecoli', 'subtilis', 'desulfuricans', 'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    
    dirs = ['hmms', 'hmmtext']
    dirs += ["hmmtext/%.5s" % (zsname) for zsname in zsnames]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def runmetrics(predicted, actual, setname, machine):
    # CLASSES ARE:
    #
    # 0: unpaired
    # 1: paired
    # 
    #print(predicted.shape, actual.shape)
    
    # predicted & actual from RNN is an np array of shape (samples, timesteps, classes): probability predictions
    if machine == 'RNN':
        y = np.argmax(np.concatenate([actual, 1 - np.sum(actual, axis=-1, keepdims=True)], axis=-1), axis=-1)
        #y = np.argmax(actual, axis = -1)
        yhat = np.argmax(predicted, axis = -1)
    
    
    
    # predicted & actual from HMM is a list of np arrays of length (timesteps, ): class predictions
    if machine == 'HMM':
        maxlength = max([s.shape[-1] for s in predicted])
        y = np.concatenate([np.pad(s, (0, maxlength - s.shape[-1]), 'constant', constant_values = (2,))[None, :] for s in actual], axis = 0)
        yhat = np.concatenate([np.pad(s, (0, maxlength - s.shape[-1]), 'constant', constant_values = (2,))[None,:] for s in predicted], axis = 0)
    
    print
    print('%s Metrics' % setname)
    print
    
    # actual
    unpaired = y == 0
    paired = y == 1
    
    # predicted
    negative = yhat == 0
    positive = yhat == 1
    
    
    tpos = np.logical_and(positive, paired)
    fpos = np.logical_and(positive, unpaired)
    tneg = np.logical_and(negative, unpaired)
    fneg = np.logical_and(negative, paired)
    
    
    if setname == 'Zsuzsanna Set':
        # print(metrics for each zsuzsanna sequence
        zsnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM', 'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum', 'syne', 'ecoli', 'subtilis', 'desulfuricans', 'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
        
        tp = tpos.sum(axis=1).astype(float)
        tn = tneg.sum(axis=1).astype(float)
        fp = fpos.sum(axis=1).astype(float)
        fn = fneg.sum(axis=1).astype(float)
        
        po = tp + fp
        ne = tn + fn
        tr = tp + tn
        fa = fp + fn
        total = (tp + fn + tn + fp)
        acc = (tp + tn)/total
        ppv = tp/(tp + fp + 10e-5)
        sen = tp/(tp + fn + 10e-5)
        
        results = np.stack([total,tr,fa,tp,tn,fp,fn,tp/total,tn/total,fp/total,fn/total,acc,ppv,sen]).T
        
        print("          NAME  | TOTAL  |  TRUE  FALSE  | TPOS  TNEG  FPOS  FNEG  |   TPOS    TNEG    FPOS    FNEG  |    ACC    PPV    SEN")
        for i in range(fneg.shape[0]):
            print('%14s  |  %4d  |  %4d   %4d  | %4d  %4d  %4d  %4d  |  %.3f   %.3f   %.3f   %.3f  |  %.3f  %.3f  %.3f' % ((zsnames[i],) + tuple(results[i])))
        print
    
    tp = tpos.sum()
    fn = fneg.sum()
    tn = tneg.sum()
    fp = fpos.sum()
    
    total = tp + fn + tn + fp
    
    print('Of %d nucleotides:\n' % (total))
    
    print('%d true positives (%.2f%%)' % (tp, (float(tp)/total)*100))
    print('%d true negatives (%.2f%%)' % (tn, (float(tn)/total)*100))
    print('%d false positives (%.2f%%)' % (fp, (float(fp)/total)*100))
    print('%d false negatives (%.2f%%)\n' % (fn, (float(fn)/total)*100))
    
    print('%d true (%.2f%%)' % (tp+tn, ((float(tp)+float(tn))/total)*100))
    print('%d false (%.2f%%)\n' % (fp+fn, ((float(fp)+float(fn))/total)*100))
    
    print('ppv: %.4f' % (tp/(tp + fp + 10e-5)))
    print('sen: %.4f\n' % (tp/(tp + fn + 10e-5)))
    
    return (float(tn + tp)/total, float(fp)/total, float(fn)/total)



def outputs(seq, actu, pred, ep, foldername = ''):
    
    zsnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM', 'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum', 'syne', 'ecoli', 'subtilis', 'desulfuricans', 'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    ndict = {0 : 'A', 1 : 'C', 2 : 'G', 3 : 'U', 4 : 'X', 5 : 'E'}
    ldict = {0 : 'U', 1 : 'P', 2 : 'E'}
    
    if foldername:
        foldername = '-' + foldername
    
    # text output
    for j in range(16):
        
        textfile = open('text/%.5s/epoch%02d.txt' % (zsnames[j], ep), 'w+')
        textfile.write('Epoch %02d\n\n' % (ep))
        
        seqletter = np.argmax(seq[j], axis = -1)
        actuletter = np.argmax(actu[j], axis = -1)
        predletter = np.argmax(pred[j], axis = -1)
        
        for i in range(1562):
            printstr = '%s  %s  %s  %d\n' % (ndict[seqletter[i]], ldict[actuletter[i]], ldict[predletter[i]], actuletter[i] == predletter[i])
            textfile.write(printstr)
        textfile.write('\n')
        textfile.close()



def hmmoutputs(seq, actu, pred, k, foldername = ''):
    
    zsnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM', 'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum', 'syne', 'ecoli', 'subtilis', 'desulfuricans', 'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    ndict = {0 : 'A', 1 : 'C', 2 : 'G', 3 : 'U', 4 : 'X', 5 : 'E'}
    ldict = {0 : 'U', 1 : 'P', 2 : 'E'}
    
    if foldername:
        foldername = '-' + foldername
    
    # text output
    for s, a, p, zsname in zip(seq, actu, pred, zsnames):
        
        textfile = open('hmmtext/%.5s/epoch%02d.txt' % (zsname, k), 'w+')
        textfile.write('HMM, k = %d\n\n' % (k,))
        
        
        for i in range(s.shape[0]):
            printstr = '%s  %s  %s  %d\n' % (ndict[s[i]], ldict[a[i]], ldict[p[i]], a[i] == p[i])
            textfile.write(printstr)
        textfile.write('\n')
        textfile.close()
        
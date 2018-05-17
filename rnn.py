import argparse
import numpy as np

import os
os.environ['THEANO_FLAGS'] = 'device=cuda0,floatX=float32'
os.environ['CPLUS_INCLUDE_PATH'] = '/usr/local/cuda/include'

import keras as k
import tools
import makebatches
import sys

from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Dropout, Dense, Conv1D, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l1,l2
from keras.optimizers import Adam, RMSprop
from time import time

np.random.seed(554433)

# command line args
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default = 50, type = int)
parser.add_argument("--batchsize", default = 50, type = int)
parser.add_argument("--samples", default = 0, type = int) # 0 = largest permissible
parser.add_argument("--timesteps", default = 0, type = int) # 0 = max of input dataset
parser.add_argument("--testprop", default = 0.5, type = float)
parser.add_argument("--reg", default = 0.0001, type = float)
parser.add_argument("--regtype", default = 'l1', type = str)
parser.add_argument("--dropprob", default = 0.5, type = float)
parser.add_argument("--hiddensizes", default = [200, 50], nargs='+', type = int)
parser.add_argument("--convsize", default = 50, type = int)
parser.add_argument("--kernelsize", default = 30, type = int)
parser.add_argument("--verbose", default = 1, type = int)
parser.add_argument("--outputs", default=True, type = bool)
parser.add_argument("--lr", default= 0.0001, type=float)
parser.add_argument("--optimizer", default = 'rmsprop')
parser.add_argument("--load", default=False, type = bool)
parser.add_argument("--lrdecay", default= 1., type=float)
parser.add_argument("--loadfile", default= '', type=str)

args = parser.parse_args()

# hyperparameters
epochs = args.epochs
batchsize = args.batchsize
samples = args.samples # number of sequences to use
timesteps = args.timesteps # timesteps to use; cut off everything after this
datadim = 5 # ACGUX
testproportion = args.testprop # proportion of set to use as validation
reg = args.reg
if args.regtype == 'l2':
    regularizer = l2(reg)
if args.regtype == 'l1':
    regularizer = l1(reg)
dropprob = args.dropprob
optim = args.optimizer
loadfile = args.loadfile
kernelsize = args.kernelsize
convsize = args.convsize
hiddensizes = args.hiddensizes

if timesteps == 0:
    timesteps = None

# load in data
#crwdata = np.load("data/crw16s.npy")
#crwdata = np.random.permutation(crwdata) # shuffle samples

#testsize = int(samples*testproportion)
#trainsize = samples - testsize

## break into X/Y, train/test
#crwdata = crwdata[:samples,:timesteps,:]
#sequences, labels = crwdata[:,:,1:6], crwdata[:,:,7:9]

#cutindex = int(sequences.shape[0]*(1 - testproportion))
#X, Xtest = sequences[:cutindex], sequences[cutindex:]
#Y, Ytest = labels[:cutindex], labels[cutindex:]



## load test sets
#testsetpaths = ['data/zs.npy', 'data/crw23s.npy', 'data/crw5s.npy']
#testsetnames = ['Zsuzsanna Set', '23S Set', '5S Set']

##testsets = [(X, Y, 'Training Set', X.shape[0]), (Xtest, Ytest, 'Validation Set', Xtest.shape[0])]
#testsets = [(Xtest, Ytest, 'Validation Set')]

#for path, name in zip(testsetpaths, testsetnames):
    #testset = np.load(path)
    #testsets.append((testset[:,:,1:6], testset[:,:,7:9], name))

#print([testset[2] for testset in testsets])
#print([testset[1].shape for testset in testsets])
    
if args.load:
    if not loadfile:
        loadfile = 'rnnfinal.h5'
    
    try:
        model = load_model(loadfile)
    except Exception:
        print('Cannot find loadfile %s' % (loadfile,))
        quit()
    
    for testset, testsetlabels, setname in testsets:
        testpredictions = model.predict(testset, batch_size = batchsize, verbose = args.verbose)
        tools.runmetrics(testpredictions, testsetlabels, setname = setname, machine = "RNN")
        if setname == 'Zsuzsanna Set':
            tools.outputs(testset, testsetlabels, testpredictions, 70)
    
    quit()


# make model
model = Sequential()

lstmargdict = {'return_sequences':True, 'kernel_regularizer':regularizer,
               'recurrent_regularizer':regularizer, 'dropout':dropprob, 'implementation':2}

convargdict = {'kernel_size': kernelsize, 'strides':1, 'activation':'relu',
               'padding':'same', 'kernel_regularizer':regularizer}

model.add(Conv1D(filters = convsize, input_shape = (None, datadim), **convargdict))
#model.add(Conv1D(filters = 50, **convargdict))
#model.add(BatchNormalization())

for h in hiddensizes:
    model.add(Bidirectional(LSTM(h, **lstmargdict)))

#model.add(BatchNormalization())
#model.add(Conv1D(filters = 50, **convargdict))
convargdict.update({'activation':'softmax'})
model.add(Conv1D(filters = 2, **convargdict))

if optim == 'adam':
    opt = Adam(lr = args.lr)
elif optim == 'rmsprop':
    opt = RMSprop(lr = args.lr)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[])

print(model.summary())

# print machine description
#layers = [l.layer if 'layer' in l.get_config() else l for l in model.layers] # for bidirectional
#tools.printstart(layers, trainsize, testsize, batchsize, timesteps, datadim, epochs, reg, dropprob, args.lr, args.lrdecay, optim)

# make output directories
tools.makernndirs()

# set up array print options
np.set_printoptions(precision=6, threshold=10000, suppress=True)

# set up training, time variables
trainsize = makebatches.findsize('data/crw16s-filtered.txt')
batchgenerator = makebatches.batch_generator('data/crw16s-filtered.txt', batchsize, length = timesteps)

testset_x, testset_y = makebatches.makebatch('data/zs.txt', 16, np.arange(16), 16)

acc = []
t1 = time()

for i in range(epochs):
    print('----------------------\n')
    print('Training Epoch %d\n' % (i+1))
    
    # train
    for j in range(trainsize//batchsize):
        batch_x, batch_y = next(batchgenerator)
        loss = model.train_on_batch(batch_x, batch_y)
        print(trainsize//batchsize, j, loss)
    
    epochaccs = []
    
    testset_yhat = model.predict(testset_x)
    epochaccs.append(tools.runmetrics(testset_yhat, testset_y, setname = "Zsuzsanna Set", machine = "RNN"))
    
    acc.append(epochaccs)
    
    #if args.outputs:
        #tools.outputs(ZX[:16], ZY, ZYhat, i+1)
    
    print('(%4.1f seconds)\n\n' % (time() - t1))
    
    #if args.lrdecay != 1:
        #print("Changing lr: %f" % (opt.lr.get_value())),
        #opt.lr.set_value(np.float32(opt.lr.get_value()*(args.lrdecay)))
        #print("to %f\n" % (opt.lr.get_value()))
    
    model.save("rnns/rnnepoch%02d.h5" % (i+1))
    
    sys.stdout.flush()

print
for i in acc:
    print(('(%02.3f %02.3f %02.3f) '*len(i)) % tuple([j for k in i for j in k]))

model.save("rnnfinal.h5")
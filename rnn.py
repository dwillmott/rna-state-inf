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

convargdict = {'kernel_size': kernelsize, 'strides':1, 'activation':'relu',
               'padding':'same', 'kernel_regularizer':regularizer}

lstmargdict = {'return_sequences':True, 'kernel_regularizer':regularizer,
               'recurrent_regularizer':regularizer, 'dropout':dropprob, 'implementation':2}

model.add(Conv1D(filters = convsize, input_shape = (None, datadim), **convargdict))

for h in hiddensizes:
    model.add(Bidirectional(LSTM(h, **lstmargdict)))

convargdict.update({'activation':'softmax'})
model.add(Conv1D(filters = 2, **convargdict))

if optim == 'adam':
    opt = Adam(lr = args.lr)
elif optim == 'rmsprop':
    opt = RMSprop(lr = args.lr)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[])

print(model.summary())

# make output directories
tools.makernndirs()

# set up training, time variables
trainsize = makebatches.findsize('data/crw16s-filtered.txt')
batchgenerator = makebatches.batch_generator('data/crw16s-filtered.txt', batchsize, length = timesteps)

testset_x, testset_y = makebatches.makebatch('data/testset.txt', 16, np.arange(16), 16)

metrics = []
t1 = time()

for i in range(epochs):
    print('----------------------\nTraining Epoch %d\n' % (i+1))
    
    # train
    #for j in range(trainsize//batchsize):
        #batch_x, batch_y = next(batchgenerator)
        #loss = model.train_on_batch(batch_x, batch_y)
        #if j % 10 == 0 and args.verbose:
            #print(trainsize//batchsize, j, loss)
    
    # predict on test set, print results
    testset_yhat = model.predict(testset_x)
    epochmetrics = tools.runmetrics(testset_y, testset_yhat, setname = "Test Set", machine = "RNN")
    metrics.append(epochmetrics)
    
    # write predictions to file
    if args.outputs:
        tools.writeoutput(testset_x, testset_y, testset_yhat, machine = "RNN", epoch = i+1)
    
    model.save("rnns/rnnepoch%02d.h5" % (i+1))
    print('(%4.1f seconds)\n\n' % (time() - t1))
    sys.stdout.flush()

print
for i in acc:
    print(('(%02.3f %02.3f %02.3f) '*len(i)) % tuple([j for k in i for j in k]))

model.save("rnnfinal.h5")
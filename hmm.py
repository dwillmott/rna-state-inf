import numpy as np
import sys
import itertools
from time import time
import tools
import os
#from processdata import *
from makebatches import getallsamples

class HMM:
    def __init__(self, nobs, order):
        self.nstates = 2
        self.nobs = nobs
        self.order = order
        self.A = np.zeros([self.nstates]*(self.order+1))
        self.B = np.zeros([self.nobs] + [self.nstates]*self.order)
        
    def train(self, states, obs, verbose = True):
        for o, s in zip(obs, states):
            #print(o)
            #print(s)
            for i in range(self.order,len(o)):
                indices = s[i-self.order:i+1]
                indices = tuple(indices[::-1])
                
                self.A[indices] += 1
                self.B[(o[i],) + indices[:-1]] += 1
        
        if verbose:
            self.printtransitions()
        
        self.A = self.A.astype(float) / (self.A.sum(axis=0)+1)
        self.B = self.B.astype(float) / (self.B.sum(axis=0)+1)
        
        if verbose:
            self.printtransitions()
        
        self.A = np.log(self.A)
        self.B = np.log(self.B)
        
        if verbose:
            self.printtransitions()
        
        return
    
    
    def printtransitions(self):
        print(self.A, '\n', self.B, '\n')
        return
    
    
    def predict(self, o):
        self.v = np.zeros([len(o)] + [self.nstates]*self.order)
        self.ptr = np.zeros([len(o)] + [self.nstates]*self.order).astype(int)
        self.v[0:self.order,:] = np.log(1./self.v[0,:].size) # begin in random starting state
        
        for t in range(self.order, len(o)):
            for indices in itertools.product(range(self.nstates), repeat=self.order):
                self.v[(t,) + indices] = self.B[(o[t],) + indices] + np.max(self.v[(t-1,) + indices[1:]] + self.A[indices])
                self.ptr[(t,) + indices] = int(np.argmax(self.v[(t-1,) + indices[1:]] + self.A[indices]))
        
        
        out = list(np.unravel_index(np.argmax(self.v[-1,:]),self.v[-1,:].shape))
        for t in range(len(o)-1,self.order-1,-1):
            out.append(np.unravel_index(self.ptr[(t,) + tuple(out[-self.order:])], self.ptr[t,:].shape)[-1])
        
        out.reverse()
        
        return np.array(out)
    
    
    def predictset(self, obs, states, setname, outputs = True):
        
        preds = [self.predict(ob) for ob in obs]
        tools.runmetrics(preds, states, setname, machine="HMM")
        
        if outputs:
            tools.hmmoutputs(obs, states, preds, self.order)
        return preds
    
    
    def save(self, path):
        np.save(path + "A.npy", self.A)
        np.save(path + "B.npy", self.B)


#def getsequences(path):
    
    #letters = 'ACGU'
    #sequences = loadsequences(path)
    #obs = [np.array([letters.index(i) if i in letters else 4 for i in seq.sequence]) for seq in sequences]
    
    #states= [seq.state for seq in sequences]
    
    #return obs, states

def getsequences(path):
    # gets sequences and states, returns them as lists of numpy arrays
    sequences, states = getallsamples(path)
    obs = [np.array(sequence).astype(int) - 1 for sequence in sequences]
    states = [np.array(state).astype(int) for state in states]
    
    return obs, states



if __name__ == "__main__":
    
    np.set_printoptions(threshold=100000, linewidth=300, precision=3)
    
    tools.makehmmdirs()
    
    assert len(sys.argv) >= 3, 'needs 2 arguments: mode and order'
    
    # two command line arguments: mode and order
    mode = sys.argv[1]
    k = int(sys.argv[2])
    
    # load training & zs sets
    crwobs, crwstates = getsequences("data/crw16s-filtered.txt")
    zobs, zstates = getsequences("data/zs.txt")
    
    print(len(crwobs))
    
    
    if mode == "train":
        H = HMM(5, k)
        H.train(crwstates, crwobs, verbose = False)
        H.save("hmms/hmmorder%d" % (k,))
        
    if mode == "run":
        H = HMM(5, k)
        H.A = np.load("hmms/hmmorder%dA.npy" % (k,))
        H.B = np.load("hmms/hmmorder%dB.npy" % (k,))
        
        #H.predictset(crwobs, crwstates, "Training Set")
        H.predictset(zobs, zstates, "Zsuzsanna Set")
        
    if mode == "cycle":
        for i in range(1, k+1):
            
            H = HMM(5, i)
            print("\n\nHMM with %d previous states \n\n" % (i,))
            
            try:
                H.A = np.load("hmms/hmmorder%dA.npy" % (i,))
                H.B = np.load("hmms/hmmorder%dB.npy" % (i,))
            except IOError:
                print("Count not find order %d HMM. Training:" % (i,))
                t = time()
                H.train(crwstates, crwobs, verbose = False)
                print("\nTrain time: %.1f seconds \n" % (time() - t))
                H.save("hmms/hmmorder%d" % (i,))
            t = time()
            H.predictset(crwobs, crwstates, "Training Set")
            H.predictset(zobs, zstates, "Zsuzsanna Set")
            print("\nTotal time: %.1f seconds \n" % (time() - t))
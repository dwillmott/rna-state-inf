import os
import sys
import subprocess


# takes output from LSTM and creates a .shape file from the predictions
def makeshape(shapetype, a = 0.214, b = 0.6624, probfile, outfile = None):
    s = 0.3603
    pairdict = {'U' : 0.0, 'P' : 1.0}
    
    outfolder = 'shape/%s/' % (shapetype,)
    if outfile == None:
        outfile = outfolder + probfile[14:-4] + '.shape'

    if not os.path.exists(resultfolder):
        os.system('mkdir ' + resultfolder)
    
    for zs, testlength in zip(testnames, testlengths):
        probfile = 'probabilities/%s-prob.txt' % (zs,)
        name = zs
        
        #read in probability file, ignore first two lines
        prob_file = open(probfile, 'r')
        prob_file.readline()
        prob_file.readline()

        seq_index = []
        lstm_prob = []
        
        for line in prob_file.readlines():
            if line != '\n':
                splitline = line.split()
                seq_index.append(int(splitline[0]))
                
                if shapetype == 'nativestate':
                    lstm_prob.append(pairdict[splitline[2]]) # look at native pair 'U' or 'P', convert to float
                if shapetype == 'predictedstate':
                    lstm_prob.append(float(splitline[5])) # use outputted probability
        
        prob_file.close();

        #open output file
        outfile = name + '-%s.shape' % (shapetype,)
        output_file = open(resultfolder+outfile, 'w')

        #for each position, write a SHAPE value depending on the probability
        for i in range(testlength):
            if lstm_prob[i] >= 0.5:
                shapevalue = ((a-s)/0.5)*(lstm_prob[i]-1)+a
            else:
                shapevalue = ((s-b)/0.5)*lstm_prob[i]+b
            output_file.write('%d %.3f \n' % (seq_index[i], shapevalue))

        output_file.close()

# takes two .ct files and returns the set of base pairs
def getpairs(structfile):
    real_struct_file = open(structfile, 'r');

    pairs = []

    firstline = real_struct_file.readline()

    for line in real_struct_file.readlines():
        splitline = line.split()
        position = int(splitline[0])
        pairedwith = int(splitline[4])
        if pairedwith and position < pairedwith:
            pairs.append((position, pairedwith))
    
    return set(pairs)

# takes two sets of base pairs and returns PPV, sen, accuracy
def getmetrics(native, predicted, name = None):
    tp = native.intersection(predicted)
    fn = native.difference(predicted)
    fp = predicted.difference(native)
    
    PPV = len(tp)/float(len(predicted))
    sen = len(tp)/float(len(native))
    accuracy = 0.5*(PPV + sen)
    
    return PPV, sen, accuracy

# 
def comparects(tocompare):
    
    print('\n\n     %s direction     \n\n' % (tocompare))

    for testname in testnames:
        print('--------  %s  --------\n' % (testname,))
        native_pairs = getpairs('nativestructures/%s-native-nop.ct' % (testname,))
        comparison_pairs = getpairs('structures/%s/%s-%s.ct' % (tocompare, testname, tocompare))
        metrics = getmetrics(native_pairs, comparison_pairs, tocompare)
        print('PPV: %0.3f     sen: %0.3f     acc: %0.3f\n' % tuple(metrics))

# takes a sequence and shape type and performs (directed) NNTM using gtfold
def makestructures(shapetype):
    filepath = 'structures/%s' % (shapetype,)
    shapefolder = 'shape/%s' % (shapetype,)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for testname in testnames:
        runstring = 'gtmfe sequences/%s-sequence.txt ' % (testname,)
        outputstring = '--output %s/%s-%s' % (filepath, testname, shapetype)
        
        print('\n\nmaking %s structure for %s \n\n' % (shapetype, testname))
        
        if shapetype == 'noshape':
            subprocess.call(runstring + outputstring, shell=True)
        else:
            shapestring = ' --useSHAPE %s/%s-%s.shape' % (shapefolder, testname, shapetype)
            subprocess.call(runstring + outputstring + shapestring, shell=True)



if __name__ == '__main__':
    
    for direc in ['shape', 'structures', 'results']:
        if not os.path.exists(direc):
            os.makedirs(direc)
    
    testnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM',
           'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
           'syne', 'ecoli', 'subtilis', 'desulfuricans',
           'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    
    testprobabilities = ['probabilities/%s-prob.txt' % (seq,) for seq in testnames]

    testlengths = [1295, 1244, 697, 1437,
             1486, 1493, 956, 1519,
             1488, 1542, 1553, 1551,
             1474, 1562, 1503, 1474]
    
    for shapetype in ['nativestate', 'predictedstate']:
        makeshape(shapetype)
    
    for shapetype in ['noshape', 'nativestate', 'predictedstate']:
        makestructures(shapetype)
    
    for shapetype in ['noshape', 'nativestate', 'predictedstate']:
        comparects(shapetype)
    
    
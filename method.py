import os
import sys
import subprocess


# takes output from LSTM and creates a .shape file from the predictions
def makeshape(shapetype, a = 0.214, b = 0.6624, outfile = None):
    s = 0.3603
    
    outfolder = 'shape/%s/' % (shapetype,)

    if not os.path.exists(outfolder):
        os.system('mkdir ' + outfolder)
    
    for testname in testnames:
        # open probability file, ignore first two lines
        probfile = 'probabilities/%s-stateprediction.txt' % (testname,)
        prob_file = open(probfile, 'r')
        prob_file.readline()
        prob_file.readline()

        # read each line, add probability to lists
        ntides = []
        lstm_probs = []
        for line in prob_file.readlines():
            if line != '\n':
                splitline = line.split()
                ntides.append(splitline[1])
                if shapetype == 'nativestate':
                    lstm_probs.append(float(splitline[2])) # use native state (0 or 1)
                if shapetype == 'predictedstate':
                    lstm_probs.append(float(splitline[4])) # use outputted probability
        
        prob_file.close()

        #open output shape file
        outfile = testname + '-%s.shape' % (shapetype,)
        output_file = open(outfolder+outfile, 'w')

        #for each position, write a SHAPE value depending on the probability
        for i, lstm_prob in enumerate(lstm_probs):
            if lstm_prob >= 0.5:
                shapevalue = ((a-s)/0.5)*(lstm_prob-1)+a
            else:
                shapevalue = ((s-b)/0.5)*lstm_prob+b
            output_file.write('%d %.3f \n' % (i+1, shapevalue))
        
        output_file.close()
        
        # write sequence to file for gtfold
        sequencefile = open('sequences/%s-sequence.txt' % (testname,), 'w')
        sequence = ''.join(ntides)
        sequencefile.write(sequence)
        sequencefile.close()

    return

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
    
    return

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

# takes two sets of base pairs and returns PPV, sensitivity, accuracy
def getmetrics(native, predicted):
    tp = native.intersection(predicted)
    fn = native.difference(predicted)
    fp = predicted.difference(native)
    
    PPV = len(tp)/float(len(predicted))
    sen = len(tp)/float(len(native))
    accuracy = 0.5*(PPV + sen)
    
    return PPV, sen, accuracy

# write PPV, sen, acc for each file to results directory
def comparects(tocompare):
    f = open('results/%s-results.txt' % (tocompare), 'w')
    
    f.write('\n     %s %s direction     \n\n\n' % (tocompare[:-5], tocompare[-5:]))

    for testname in testnames:
        f.write('--------  %s  --------\n\n' % (testname,))
        native_pairs = getpairs('nativestructures/%s-native-nop.ct' % (testname,))
        comparison_pairs = getpairs('structures/%s/%s-%s.ct' % (tocompare, testname, tocompare))
        metrics = getmetrics(native_pairs, comparison_pairs)
        f.write('PPV: %0.3f     sen: %0.3f     acc: %0.3f\n\n' % tuple(metrics))
    
    f.close()
    return


if __name__ == '__main__':
    
    for direc in ['sequences', 'shape', 'structures', 'results']:
        if not os.path.exists(direc):
            os.makedirs(direc)
    
    testnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM',
           'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
           'syne', 'ecoli', 'subtilis', 'desulfuricans',
           'reinhardtiiC', 'maritima', 'tenax', 'volcanii']
    
    testprobabilities = ['probabilities/%s-stateprediction.txt' % (seq,) for seq in testnames]
    
    for shapetype in ['nativestate', 'predictedstate']:
        makeshape(shapetype)
    
    for shapetype in ['noshape', 'nativestate', 'predictedstate']:
        makestructures(shapetype)
    
    for shapetype in ['noshape', 'nativestate', 'predictedstate']:
        comparects(shapetype)
    
    
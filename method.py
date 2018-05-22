import os
import sys
import subprocess


def getprobability(x):
    s = 0.3603
    if x >= 0.5:
        shape = ((a-s)/0.5)*(x-1)+a
    if x < 0.5:
        shape = ((s-b)/0.5)*x+b
    
    return shape


def makeshape(functiontype, a = 0.214, b = 0.6624):
    s = 0.3603
    
    resultfolder = 'shape/'+functiontype+'/'

    if not os.path.exists(resultfolder):
        os.system('mkdir ' + resultfolder)
    
    for zs, zslength in zip(zsnames, zslengths):
        probfile = 'probabilities/%s-prob.txt' % (zs,)
        name = zs
        
        #read in probability file
        prob_file = open(probfile, 'r')
        
        prob_file.readline()
        prob_file.readline()

        seq_index = []
        lstm_prob = []
        
        pairdict = {'U' : 0.0, 'P' : 1.0}
        
        for line in prob_file.readlines():
            if line != '\n':
                splitline = line.split()
                seq_index.append(int(splitline[0]))
                
                if functiontype == 'nativestate':
                    lstm_prob.append(pairdict[splitline[2]]) # look at native pair 'U' or 'P', convert to float
                if functiontype == 'predictedstate':
                    lstm_prob.append(float(splitline[5])) # use outputted probability
        
        
        prob_file.close();

        #open output file
        outfile = name + '-%s.shape' % (functiontype,)
        output_file = open(resultfolder+outfile, 'w')
        

        #for each position, write a SHAPE value depending on the probability 
        for i in range(zslength):
            #shapevalue = getprobability(lstm_prob[i])
            shapevalue = ((a-s)/0.5)*(lstm_prob[i]-1)+a if lstm_prob[i] >= 0.5 else ((s-b)/0.5)*lstm_prob[i]+b
            position = seq_index[i];
            output_file.write('%d %.3f \n' % (position, shapevalue))

        output_file.close()


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


def getmetrics(native, predicted, name = None):
    tp = native.intersection(predicted)
    fn = native.difference(predicted)
    fp = predicted.difference(native)
    
    PPV = len(tp)/float(len(predicted))
    sen = len(tp)/float(len(native))
    accuracy = 0.5*(PPV + sen)
    
    return PPV, sen, accuracy


def comparects(tocompare):
    
    print('\n\n     %s direction     \n\n' % (tocompare))

    for zsname in zsnames:
        print('--------  %s  --------\n' % (zsname,))
        native_pairs = getpairs('nativestructures/%s-native-nop.ct' % (zsname,))
        comparison_pairs = getpairs('structures/%s/%s-%s.ct' % (tocompare, zsname, tocompare))
        metrics = getmetrics(native_pairs, comparison_pairs, tocompare)
        print('PPV: %0.3f     sen: %0.3f     acc: %0.3f\n' % tuple(metrics))


def makestructures(subfolder):
    filepath = 'structures/%s' % (subfolder,)
    shapefolder = 'shape/%s' % (subfolder,)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for zsname in zsnames:
        runstring = 'gtmfe sequences/%s-sequence.txt ' % (zsname,)
        outputstring = '--output %s/%s-%s' % (filepath, zsname, subfolder)
        
        print('\n\nmaking %s structure for %s \n\n' % (subfolder, zsname))
        
        if subfolder == 'noshape':
            subprocess.call(runstring + outputstring, shell=True)
        else:
            shapestring = ' --useSHAPE %s/%s-%s.shape' % (shapefolder, zsname, subfolder)
            subprocess.call(runstring + outputstring + shapestring, shell=True)



filenames = ['noshape-results.txt',
             'nativestate-results.txt',
             'predictedstate-results.txt']

if __name__ == '__main__':
    
    for direc in ['shape', 'structures', 'results']:
        if not os.path.exists(direc):
            os.makedirs(direc)
    
    zsnames = ['cuniculi', 'vnecatrix', 'celegans', 'nidulansM',
           'TabacumC', 'cryptomonasC', 'musM', 'gallisepticum',
           'syne', 'ecoli', 'subtilis', 'desulfuricans',
           'reinhardtiiC', 'maritima', 'tenax', 'volcanii']

    zslengths = [1295, 1244, 697, 1437,
             1486, 1493, 956, 1519,
             1488, 1542, 1553, 1551,
             1474, 1562, 1503, 1474]
    
    for shapetype in ['nativestate', 'predictedstate']:
        makeshape(shapetype)
    
    for shapetype in ['noshape', 'nativestate', 'predictedstate']:
        makestructures(shapetype)
    
    for shapetype in ['noshape', 'nativestate', 'predictedstate']:
        comparects(shapetype)
    
    
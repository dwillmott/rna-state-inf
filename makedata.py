import numpy as np
import glob

sequencedict = {'A' : '1', 'C' : '2', 'G' : '3', 'U' : '4'}
for letter in 'BDEFHIJKLMNOPQRSTVWXYZ':
    sequencedict.update({letter : '5'})


def getsequenceandstructure(filename, headersize):
    data = np.loadtxt(filename, skiprows = headersize, dtype='str')
    
    sequence = data[:,1]
    sequence = [sequencedict[s.upper()] for s in sequence]
    structure = data[:,4]
    state = structure.astype(bool).astype(int).astype(str)
    
    return sequence, structure, state

def writedatafile(paths, outfile, headersize):
    
    f = open(outfile, 'w')
    
    for path in paths:
        sequence, structure, state = getsequenceandstructure(path, headersize)
        f.write(path + '\n')
        f.write(' '.join(sequence) + ' \n')
        f.write(' '.join(structure) + ' \n')
        f.write(' '.join(state) + ' \n')
        f.write('\n')

    f.close()
    return

if __name__ == '__main__':
    
    globstring = 'data/raw/crw16s/**/*.nopct'
    paths = glob.glob(globstring, recursive = True)
    outfile = 'data/crw16s-comparative.txt'
    headersize = 5
    
    
    writedatafile(paths, outfile, headersize)
    
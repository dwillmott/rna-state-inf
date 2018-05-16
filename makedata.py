import numpy as np
import glob


sequencedict = {'A' : '0', 'C' : '1', 'G' : '2', 'U' : '3'}
for letter in 'BDEFHIJKLMNOPQRSTVWXYZ':
    sequencedict.update({letter : '4'})


def getsequenceandstructure(filename, headersize):
    data = np.loadtxt(filename, skiprows = headersize, dtype='str')
    
    sequence = data[:,1]
    sequence = [sequencedict[s.upper()] for s in sequence]
    structure = data[:,4]
    state = structure.astype(bool).astype(int).astype(str)
    
    return sequence, structure, state

# convert list of .ct files into a textfile; each .ct file is four lines:
#
# filename
# sequence (A=0, C=1, G=2, U=3, all else = 4)
# structure
# state (unpaired=0, paired=1)
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
    
    # CHANGE THESE IF YOU'RE USING YOUR OWN DATA
    datadirectory = 'rawdata'  # directory with .ct files
    outfile = 'data/crw5s-comparative.txt'  # output file to write to
    headersize = 5  # number of lines in the .ct file before the sequence begins
    
    
    # get all filepaths
    ctglobstring = datadirectory + '/**/*.ct'
    nopctglobstring = datadirectory + '/**/*.nopct'
    paths = glob.glob(ctglobstring, recursive = True) + glob.glob(nopctglobstring, recursive = True)
    
    writedatafile(paths, outfile, headersize)
    
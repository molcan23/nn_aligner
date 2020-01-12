import sys
import os
import glob
import tempfile
import argparse
import numpy as np
import h5py
import mappy as mp
import re

out_header = "# <sequenceFileName> <reference_begin> <reference_end> <sequence_begin> <sequence_end>"

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--referenceFile", help="location of reference .fa file", type=str)
parser.add_argument("-s", "--sequenceFolder", help="folder with sequence files", type=str)
parser.add_argument("-mm", "--minimalMatch", help="minimal size of match that will can be on output", type=int, default = 50)
parser.add_argument("-o", "--outputFile", help="specifies output file", type=str, default = "out.txt")
parser.add_argument('-raw', action='store_true', help="output raw signal")
parser.add_argument('-fake', action='store_true', help="create fake read from hit")


class Table_Iterator:

    def __init__(self, basecallEventTable):
        self.table = basecallEventTable
        self.tableindex = 0
        self.localindex = 0
    def __iter__(self):
        return self

    def __next__(self):

        while self.localindex == 5:
            if self.tableindex + 1 != len(self.table):
                self.tableindex += 1
                self.localindex = 5-int(self.table[self.tableindex][5])
            else:
                raise StopIteration

        self.localindex += 1
        return self.table[self.tableindex][4][self.localindex-1]

################################################################################

args = parser.parse_args()

assert os.path.isfile(args.referenceFile), "Reference file not exists."

# recursively find all fast5 files in directory

fast5Files = glob.glob(args.sequenceFolder + '/**/*.fast5', recursive=True)
print(args.referenceFile.split('/')[-1][:-3])
# out_name = args.referenceFile.split('/')[-1][:-3] + '_to_basecalled.txt'
out_name = args.outputFile
print("Found %d .fast5 files\n" % (len(fast5Files)))


# create fasta file from sequence strings

fastaSequenceFile = tempfile.NamedTemporaryFile(mode = 'w', suffix = '.fa', delete = False)

for file in fast5Files:
    
    sequenceFile = h5py.File(file, 'r')
    basecallOut = str(sequenceFile['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'][()]).split('\\n')
    basecallString = basecallOut[1]
    fastaSequenceFile.write(">" + file + "\n")
    fastaSequenceFile.write(basecallString + "\n")
    sequenceFile.close()

fastaSequenceFile.close()

# index reference File

sequenceIndex = mp.Aligner(args.referenceFile)
assert sequenceIndex, "failed to load/build reference index"

# create out-file and fill it with hits
outFile = open(out_name, "w")

# header

outFile.write(out_header + "\n\n")

for name, seq, qual in mp.fastx_read(fastaSequenceFile.name): # read a fasta sequence
        for hit in sequenceIndex.map(seq, cs = True): # traverse alignments
            # print(hit, '\n\n')
            if (hit.r_en - hit.r_st < args.minimalMatch):
                continue

            outFile.write("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n" % (name, hit. ctg, hit.r_st, hit.r_en, hit.q_st,
                                                                        hit.q_en, hit.strand,
                                                                        hit.blen, hit.mapq, hit.cs))
            outFile.write("\n")


outFile.close()

# Give at least two transcription factor names
tf_names=['STAT3', 'STAT2']


############################################################################
# get all important from randomforest
canType='LAML'

############################################################################
import os, sys
import wget
import gzip
import shutil

import re, gc
import pandas as pd

from arboreto.algo import grnboost2, genie3
from arboreto.utils import load_tf_names

# TFs should be at least 2!
dir='/home/cpdong/Desktop/demo'
os.chdir(dir)

# download and unzip file
url = 'https://tcga.xenahubs.net/download/TCGA.'+ canType + '.sampleMap/HiSeqV2.gz'
wget.download(url, 'cancer_data_TCGA.gz') 

with gzip.open('cancer_data_TCGA.gz', 'rb') as f_in:
    with open('file.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove("cancer_data_TCGA.gz")
############################################################################

df= pd.read_table("file.txt",index_col=0, sep='\t')
#ex_matrix.columns.get_values()
cols = [c for c in df.columns if c[13:15] == '03']
df=df[cols]

matrix = df.T

network = grnboost2(expression_data=matrix, tf_names=tf_names, verbose=True)

network.to_csv(canType + '_ex_GRNboost2_network.tsv', sep='\t', header=False, index=False)

# release the memory from python
#del ex_matrix, matrix, df, tf_names, network
os.remove('file.txt')
shutil.rmtree('dask-worker-space')
# end function

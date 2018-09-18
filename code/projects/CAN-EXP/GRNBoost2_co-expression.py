#############################################################################
## below is the python code
#############################################################################
# get feed2python file from R merge, for I don't merge with python

# link: https://nbviewer.jupyter.org/github/tmoerman/arboreto/blob/master/notebooks/examples/ex_01_grnboost2_local.ipynb
# required package: arboreto, pandas, sklearn
# genename couldn't be numeric!
# add 'G' for entrezgene id

dir="C:/Users/Chuanpeng/Desktop/refine"

import os, sys
import pandas as pd
os.chdir(dir)

from arboreto.algo import grnboost2, genie3
from arboreto.utils import load_tf_names

ex_matrix = pd.read_csv("feed2python.csv",index_col=0)
matrix = ex_matrix.T

# tf_names = load_tf_names("ChIPBaseV2_regNet_tcga.csv")
df= pd.read_csv("https://raw.githubusercontent.com/cpdong/public/master/data/CAN-EXP/python_regulist_encode_GRNBoost2.csv")
tf_names= df['entrezgene'].tolist()

print(matrix.head(3))

network = grnboost2(expression_data=matrix, tf_names=tf_names)

network.to_csv('ex_tcga_network.tsv', sep='\t', header=False, index=False)

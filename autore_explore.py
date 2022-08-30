'''
script to compare autoRE generated from Gordon2017 vs Bartlett 2017

the error ranking is about the same

'''
import os
import pandas as pd

# Gordon data
path = os.path.join('data','autoRE_W411.csv')
ARE = pd.read_csv(path)
ARE_mad = ARE.mad().sort_values()
ARE_mad.index = ARE_mad.index.str.strip()

# Bartlett data
path = os.path.join('data','c7cp00757d2.xlsx')
BRE = pd.read_excel(path,skiprows=1,sheet_name=1)
BRE_mad = BRE.MAD.rename(BRE.Method).sort_values()
BRE_mad.index = BRE_mad.index.str.strip()

ABRE_mad = pd.concat([ARE_mad,BRE_mad],axis=1)
ind = (~ABRE_mad.isnull()).all(axis=1)
print(ABRE_mad[ind].to_string())

lbls = ABRE_mad.index[ind]

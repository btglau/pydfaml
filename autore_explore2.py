'''
script to compare autoRE generated from Gordon2017: W411 only vs augmented

'''
import os
import pandas as pd

path = os.path.join('data','autoRE_aug.csv')
ARE = pd.read_csv(path,index_col=0)
ARE_mad = ARE.mad().sort_values()
ARE_mad.index = ARE_mad.index.str.strip()
ARE_mad.name = 'aug'

path = os.path.join('data','autoRE_W411.csv')
BRE = pd.read_csv(path,index_col=0)
BRE_mad = BRE.mad().sort_values()
BRE_mad.index = BRE_mad.index.str.strip()
BRE_mad.name = 'w411'

ABRE_mad = pd.concat([ARE_mad,BRE_mad],axis=1)

# can print out to compare, or print them separately to see how orders change
# print(ABRE_mad.to_string())
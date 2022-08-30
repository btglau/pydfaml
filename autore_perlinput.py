'''
a script to process Head-Gordon's 2017 DFA benchmarks into a format for autoRE

creates reference.txt in appropriate folders for the perl script

July 27 2022 12:55 pm

bryan lau
'''

import os
import pandas as pd

dataset = ["PlatonicTAE6","AlkAtom19","TAE140"]
dataset = ["AlkAtom19","TAE140"]
dataset = ["TAE140"]

folder = 'data'
file = 'IndValues.csv'
path = os.path.join(folder,file)
T = pd.read_csv(path,dtype=str)

# read in the TAEs:
# PlatonicTAE6
# AlkAtom19
# TAE140nonMR
# TAE140MR
## there are more than 140 TAE geometries due to individual atoms (12 atoms)
ae_keep = '|'.join(dataset)
TAE = T[T.RefNames.str.contains(ae_keep)]
# Gordon reports TAE as negative numbers. Make them positive
for c in TAE.columns[2:]:
    TAE[c] = TAE[c].astype(float)
    TAE[c] *= -1

file = 'DatasetEval.csv'
path = os.path.join(folder,file)
D = pd.read_csv(path,usecols=[0,1,2],names=['DatasetRefName','stoich','molecule'])
DAE = D[D.DatasetRefName.str.contains(ae_keep)]

# prune duplicates from AlkAtom and TAE
# methane
# ethane
# propane
if 'AlkAtom19' in ae_keep:
    TAE = TAE.drop(labels=[294,295,296])
    DAE = DAE.drop(labels=[294,295,296])

# platonic C cages TAE were inputed as positive values
if 'Platonic' in ae_keep:
    TAE.iloc[-6:,2:] *= -1

# create molecular entry template
blank = 'xxxx x x x x 0 0 0 0 0'
template = []
for m in DAE.molecule:
    M = pd.read_csv(os.path.join(folder,'Geometries',m + '.xyz'),
                    sep=' ',skiprows=2,usecols=[0],names=['atoms'])
    vc = M.atoms.value_counts()
    if vc.size > 4:
        print(f'{m} has than four kinds of atoms!')
    line = m.split('_')[1] + ' '
    line += ' '.join(vc.index.values) + ' x' * (4-vc.index.values.size)
    line += ' '
    line += ' '.join(f'{vc.values}'[1:-1].split()) + ' x'  * (4-vc.index.values.size)
    line += ' '
    template.append(line)

# make an input file for each DFA
TAE['template'] = template
for c in TAE.columns[2:-1]:
    path = os.path.join(folder,'autoRE',c)
    os.makedirs(path, exist_ok=True)

    txt = pd.concat([pd.Series(blank),TAE['template'].str.cat(TAE[c].astype(str))])
    path = os.path.join(folder,'autoRE',c,'reference_' + '-'.join(dataset) + '.txt')
    txt.to_csv(path,index=False,header=False)

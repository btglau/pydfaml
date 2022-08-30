'''
A script to prepare the dataset for scikit

Aug 1 2022

Takes .csvs and outputs a one-hot encoding of the reactions

A vector of 2*(number of molecules) = 4 1's for reactants/products, 0's otherwise
A one-hot encoding of the best functional for each calculation

For Bartlett2017, produce one dataset with CCSD(T) and one without

'''

import os
import pandas as pd
import numpy as np

folder = 'data'

# aug, W411
for d in ['autoRE_aug.csv','autoRE_W411.csv','c7cp00757d2.xlsx']:

    # results from autoRE workup for all Gordon functionals
    path = os.path.join('data',d)
    usecols = [1,3,5,7]
    if 'xlsx' in d:
        # 9, 10 correspond to CCSD(T) results
        usecols += list(range(11,31))
        ARE = pd.read_excel(path,sheet_name=0,usecols=usecols)
        # drop the summary rows at the bottom
        ARE = ARE.drop(ARE.iloc[-3:].index)
        ARE.columns = ARE.columns.str.replace(' error','')
        dfas = ['PWPB95','wB97X','PBE']
    else:
        usecols += list(range(9,212))
        ARE = pd.read_csv(path,usecols=usecols)
        dfas = ['SCAN','SCAN0']
    ARE[ARE.columns[4:]] = ARE[ARE.columns[4:]].astype(float)

    # # categorically encode molecules
    # some TAEs in the Bartlett data set are missing from the full W411 set, because
    # presumably there are no valid reactions (140 -> 137)
    # {'ph3', 'p2', 'be2', 'p4'}
    mols = pd.unique(ARE[['A','B','C','D']].values.ravel('K'))
    mols_enc = {mols[i]:i for i in range(len(mols))}
    mols_key = {i:mols[i] for i in range(len(mols))}
    AREI = ARE[['A','B','C','D']].replace(mols_enc).astype(int).to_numpy()

    # one hot encode the reactants and products as two 1's, then combine them
    # each row should have four 1's
    reactants = np.zeros((AREI.shape[0],AREI.max()+1))
    products = np.zeros(reactants.shape)
    ix_rxn = np.arange(AREI.shape[0])
    reactants[np.c_[ix_rxn,ix_rxn],AREI[:,0:2]] = 1
    products[np.c_[ix_rxn,ix_rxn],AREI[:,2:]] = 1
    rp = np.c_[reactants,products]

    # instead of [reactants] + [products], combine them into one vector that doesn't
    # put a "spatial" emphasis on separating reactants and products
    rp = np.zeros(reactants.shape)
    rp[np.c_[ix_rxn,ix_rxn,ix_rxn,ix_rxn],AREI] = 1

    # test if the indexing worked
    tmp = np.reshape(np.where(rp)[1],AREI.shape)
    tmp[:,2:] -= AREI.max()+1
    # if, e.g AREI[147,:] = [  0, 114, 119, 107]
    #          tmp[147,:] = [  0, 114, 107, 119]
    print(f'rxn was one-hot encoded right? {np.all(np.sort(tmp,axis=1) == np.sort(AREI,axis=1))}')
    
    methods = ARE.columns[4:]
    if dfas:
        # only keep and rank the DFAs specified above
        methods = set(methods) - set(dfas)
        methods = list(methods)
        ARE = ARE.drop(columns=methods)
    methods_key = {i:ARE.columns[4:][i] for i in range(len(ARE.columns[4:]))}
    place = np.argsort(np.absolute(ARE[ARE.columns[4:]].to_numpy()))
    # best is a one hot encode of the best functional for each reaction
    best = np.zeros(place.shape)
    best[np.arange(best.shape[0]),place[:,0]] = 1
    print(f'Is place set properly? {np.all(np.where(best)[1] == place[:,0])}')
    # see what kind of reactions DFAs are best at
    # ind = np.where(place[:,0] == 21)[0]
    # ARE.iloc[ind,:4]

    place_count = pd.Series(place[:,0],name='places').value_counts().rename(methods_key)
    mad = ARE[ARE.columns[4:]].mad()
    mad.name = 'MAD'
    summary = pd.concat([place_count,mad],axis=1)
    summary = summary.sort_values(by='MAD')
    print(summary.to_string())

    path = os.path.join('scikit',d.split('.')[0])
    #np.save(path + '.X',rp)
    #np.save(path + '.Y',best)
    #np.save(path + '.Xlabel',mols_key)
    #np.save(path + '.Ylabel',methods_key)

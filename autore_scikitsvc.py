'''

scikit-learn:

SVM

'''
import os
import argparse
import sys
from itertools import groupby

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

def split_text(s):
    for k, g in groupby(s, str.isalpha):
        if k:
            yield ''.join(g)
        else: # not alpha, i.e. numeric
            yield int(''.join(g))

def getArgs(argv):
    '''
    get args from command line sys.argv[1:]

    dataset = 'autoRE_aug'
    dataset = 'autoRE_W411'
    dataset = 'c7cp00757d2'

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',help="dataset to use",default='autoRE_AlkAtom19-TAE140')
    parser.add_argument('-e',help='''encoding type
                                    0 - reactants and products, [r p]
                                    1 - encode into one vector of length r w/out separating
                                    2 - add information about atom counts''',default=1,type=int)
    parser.add_argument('--dfa',help="comma separated list of functionals",default='wB97X,wB97X-D')

    args = parser.parse_args(argv)
    args.dfa = args.dfa.split(',')

    return args

def loadData(args):

    dataset = args.d + '.csv'

    # results from autoRE workup for all Gordon functionals
    path = os.path.join('data',dataset)
    usecols = [1,3,5,7,8]
    if 'c7cp00757d2' in dataset:
        # 10, 11 correspond to CCSD(T) results
        usecols += list(range(12,31))
        ARE = pd.read_csv(path,usecols=usecols)
        ARE.columns = ARE.columns.str.replace(' error','')
    else:
        usecols += list(range(10,212))
        ARE = pd.read_csv(path,usecols=usecols)
    ARE[ARE.columns[5:]] = ARE[ARE.columns[5:]].astype(float)

    return ARE

def getData(args):
    '''
    get X y and labels from csv's
    '''
    ARE = loadData(args)
    dfas = args.dfa

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
    ix_rxn = np.arange(AREI.shape[0])
    if args.e == 0:
        products = np.copy(reactants)
        reactants[np.c_[ix_rxn,ix_rxn],AREI[:,0:2]] = 1
        products[np.c_[ix_rxn,ix_rxn],AREI[:,2:]] = 1
        rp = np.c_[reactants,products]
    elif args.e > 0:
        # instead of [reactants] + [products], combine them into one vector that doesn't
        # put a "spatial" emphasis on separating reactants and products
        rp = np.zeros(reactants.shape)
        rp[np.c_[ix_rxn,ix_rxn],AREI[:,0:2]] = -1
        rp[np.c_[ix_rxn,ix_rxn],AREI[:,2:]] = 1
    if args.e == 2:
        # add on a vector with number of atoms
        # TODO: this operation is slow, find a way to speed it up
        atoms = ARE.columns[4].split('-')
        counts = ARE[ARE.columns[4]].str.replace('-','').apply(split_text).apply(list)
        atom_counts = pd.DataFrame(np.zeros((ARE.shape[0],len(atoms)),dtype=int),columns=atoms)
        for ix in range(atom_counts.shape[0]):
            atom_counts.loc[ix,counts[ix][0::2]] = counts[ix][1::2]
        # delete atoms that aren't present
        atom_counts.drop(atom_counts.columns[atom_counts.sum() == 0],axis=1)
        rp = np.c_[rp,atom_counts]

    # try eliminating xxxx - superfluous since A -> C+D has three 1's anyways
    #rp = rp[:,1:]

    if args.e < 2:
        # test if the indexing worked
        tmp = np.reshape(np.where(rp)[1],AREI.shape)
        #tmp[:,2:] -= AREI.max()+1
        # if, e.g AREI[147,:] = [  0, 114, 119, 107]
        #          tmp[147,:] = [  0, 114, 107, 119]
        print(f'rxn was one-hot encoded right? {np.all(np.sort(tmp,axis=1) == np.sort(AREI,axis=1))}')
    
    # do not one hot encode for sklearn svc
    methods = ARE.columns[5:]
    if dfas:
        # only keep and rank the DFAs specified above
        methods = set(methods) - set(dfas)
        methods = list(methods)
        ARE = ARE.drop(columns=methods)
    methods_key = {i:ARE.columns[5:][i] for i in range(len(ARE.columns[5:]))}
    place = np.argsort(np.absolute(ARE[ARE.columns[5:]].to_numpy()))

    # see what kind of reactions DFAs are best at
    # ind = np.where(place[:,0] == 21)[0]
    # ARE.iloc[ind,:4]

    place_count = pd.Series(place[:,0],name='places').value_counts().rename(methods_key)
    mad = ARE[ARE.columns[5:]].abs().mean()
    mad.name = 'MAD'
    summary = pd.concat([place_count,mad],axis=1)
    summary = summary.sort_values(by='MAD')
    summary['zeroRule'] = summary.places/summary.places.sum()
    summary['randGuess'] = summary['zeroRule']**2

    # https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    # accepts vector of labels (not one-hot encoding)
    # todo: scale the feature columns?
    X = rp
    y = place[:,0]
    Xlabel = mols_key
    Ylabel = methods_key

    return X, y, Xlabel, Ylabel, summary, ARE

if __name__ == '__main__':

    args = getArgs(sys.argv[1:])

    X, y, Xlabel, Ylabel, summary, ARE = getData(args)

    # dual=False for n_samples > n_features
    # https://stackoverflow.com/questions/33843981/under-what-parameters-are-svc-and-linearsvc-in-scikit-learn-equivalent/33844092#33844092
    # don't use linear SVC if you can avoid it
    # clf = svm.LinearSVC(C=1,dual=False)

    # retain untouched test, X_train will be split in gridsearchcv
    X_train, X_test, y_train, y_test, ind_train, ind_test = model_selection.train_test_split(
                                        X, y, range(X.shape[0]), test_size=0.2, random_state=0, stratify=y)

    # fit the transformer to the training data, then apply that same transformation
    # to the test data - rationale is we don't see the test data coming in, so we don't
    # know the true range of the data
    if args.e == 2:
        transformer = preprocessing.MaxAbsScaler(copy=False).fit(X_train)
        transformer.transform(X_train)
        transformer.transform(X_test)

    # for when there is an imbalance of categories
    cv = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    param_grid = [
    {'kernel':['linear'], 'C':np.logspace(-2, 3, 10)},
    {'kernel':['rbf'], 'C':np.logspace(-2, 3, 10), 'gamma':np.logspace(-3, 3, 10)},
    ]
    grid_search = model_selection.GridSearchCV(svm.SVC(),
                                                param_grid,n_jobs=-1,verbose=3,cv=cv,
                                                return_train_score=True)
    grid_search.fit(X_train, y_train)

    # print the best model found
    cv_results_ = pd.DataFrame(grid_search.cv_results_)
    ind = grid_search.best_index_
    print(cv_results_.iloc[ind].to_string())

    # plot a bias-variance curve for cross validation
    train_cv = cv_results_[['param_kernel','param_C','param_gamma','mean_test_score','std_test_score','mean_train_score','std_train_score','rank_test_score']]

    # baseline performance
    print('')
    print(summary.to_string())

    # classification report
    y_pred = grid_search.predict(X_test)
    print('')
    print(classification_report(y_test, y_pred))

    # confusion matrix - rows are true labels, columns predicted
    methods = [Ylabel[i] for i in range(len(args.dfa))]
    cm = confusion_matrix(y_test, y_pred)
    cmdf = pd.DataFrame(cm,index=methods,columns=methods)
    print(cmdf.to_string())

    # bias, variance - Jcv vs Jtrain vs Jtest
    

    # outliers, error analysis
    ind = (y_pred == y_test)
    print('summary stats for correct guess')
    print(ARE.iloc[ind_test].iloc[ind,5:].abs().diff(axis=1).iloc[:,1].abs().describe())
    print('summary stats for incorrect guess')
    print(ARE.iloc[ind_test].iloc[~ind,5:].abs().diff(axis=1).iloc[:,1].abs().describe())

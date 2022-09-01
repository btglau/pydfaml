'''

scikit-learn:

gradient boosting methods

- use native categorical features instead of one-hot encoding

'''
import sys

import pandas as pd
import numpy as np

from autore_scikitsvc import getArgs, getData

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

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

    if args.a == 1:
        # no need to transform data for random forest or GBC
        pass

    categorical_mask = None
    if args.e == 0:
        # specify which are categorical
        categorical_mask = np.arange(4)

    # for when there is an imbalance of categories
    cv = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # 128 trees no more improvement
    # 'max_depth':None, 'min_samples_split':2
    param_grid = [
    {'l2_regularization':[0,1e-2,1e-1,1,10,1e2],'max_iter':[1,10,100,1000]}
    ]
    grid_search = model_selection.GridSearchCV(HistGradientBoostingClassifier(categorical_features=categorical_mask,scoring='f1'),
                                                param_grid,n_jobs=-1,verbose=3,cv=cv,
                                                return_train_score=True)
    grid_search.fit(X_train, y_train)

    # print the best model found
    print(grid_search.best_estimator_)
    cv_results_ = pd.DataFrame(grid_search.cv_results_)
    ind = grid_search.best_index_
    print(cv_results_.iloc[ind].to_string())

    # plot a bias-variance curve for cross validation
    train_cv = cv_results_[['mean_test_score','std_test_score','mean_train_score','std_train_score','rank_test_score']]

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

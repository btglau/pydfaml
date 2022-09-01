'''

unsupervised learning to explore autoRE dataset
- PCA, t-SNE, etc

- randomized PCA, kernel approximation
- MDS
- BernoulliRBM - unsupervised restricted boltzmann machine

'''
import sys
import numpy as np

import matplotlib
matplotlib.rcParams['figure.raise_window'] = False
matplotlib.rcParams["tk.window_focus"] = False
matplotlib.use("TkAgg")
print(f'backend: {matplotlib.rcParams["backend"]}')
import matplotlib.pyplot as plt
from matplotlib import ticker

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import manifold

from autore_scikitsvc import getArgs, getData

def plot_2d(points, points_color, title):
    plt.ion()
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

if __name__ == '__main__':

    args = getArgs(sys.argv[1:])
    ndfa = len(args.dfa)

    X, y, Xlabel, Ylabel, summary, ARE = getData(args)

    if args.a == 1:
        transformer = preprocessing.MaxAbsScaler(copy=False).fit(X)
        transformer.transform(X)

    if 1 == 0: 

        # PCA
        pca = PCA()
        pca.fit(X)
        print(pca.explained_variance_ratio_[:5])
        print(pca.singular_values_[:5])

        # figures
        plt.ion()
        fig, ax = plt.subplots()
        plt.ylim(0,1)
        x = np.arange(X.shape[1])
        ax.bar(x,pca.explained_variance_ratio_)
        ax2 = ax.twinx()
        ax2.plot(x,np.cumsum(pca.explained_variance_ratio_))
        ax.set_xlabel('Components')
        ax.set_ylabel('% variance explained')
        ax2.set_ylabel('Cum. variance explained')
        # hack to not freeze matplotlib?
        plt.show()

    # isomap embedding
    if 1 == 0:
        n_neighbors = 5  # neighborhood which is used to recover the locally linear structure
        n_components = 2  # number of coordinates for the manifold
        isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
        X_isomap = isomap.fit_transform(X)
        plot_2d(X_isomap, y, "Isomap Embedding")

    # t-SNE
    if 1 == 0:
        n_components = 2
        t_sne = manifold.TSNE(
            n_components=n_components,
            learning_rate="auto",
            perplexity=30,
            n_iter=250,
            init="random",
        )
        X_t_sne = t_sne.fit_transform(X)

        plot_2d(X_t_sne, y, "T-distributed Stochastic  \n Neighbor Embedding")
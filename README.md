# pydfaml

Machine learn a classifier to predict DFA for reactions auto-generated from TAE data

Generate input files for the autore script, then work up the results:

autore_perlinput.py

autore_perloutput.py

The resulting csv is worked up in a function contained in

autore_scikitsvc.py

Which also does SVC. Other methods tried are random forest (autore_scikitrf), and a MLP (autore_mlp)

Invariably, models can train to 99%+ accuracy on the train set, and about 80% on the test set, versus 60% on the zero rule, when distinguishing between wB97x and wB97x-d, i.e., if dispersion corrections are needed. However, dispersion corrects are likely not too important for this class of reactions, which focus on small molecules from the W4-11 Weizmann test set.

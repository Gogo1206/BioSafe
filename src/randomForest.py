import csv
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

files = glob.glob('tmp\\data\\*.csv')
labels = []
x = []
y = []
for filename in files:
    data = []
    with open(filename, newline='') as csvfile:
        labels.append(filename[9:-4])
        reader = list(csv.reader(csvfile, delimiter=','))
        title = reader.pop(0)
        # print(reader)
        for row in reader:
            # plt.title(filename)
            # plt.plot([1,2,3,4],[eval(row[i]) for i in range(4)], linestyle='dashed')
            x.append([eval(row[i]) for i in range(len(row))])
            y.append(labels.index(filename[9:-4]))
    # plt.show()

rf_train, rf_test, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=50)

rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf_model.fit(rf_train, train_label)
prediction_test = rf_model.predict(X=rf_test)

# Accuracy on Test
print("Training Accuracy is: ", rf_model.score(rf_train, train_label))
# Accuracy on Train
print("Testing Accuracy is: ", rf_model.score(rf_test, test_label))

#confusion matrix
# cm = confusion_matrix(test_label, prediction_test)
# cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
# plt.figure()
# plot_confusion_matrix(cm_norm, classes=rf_model.classes_)


prediction = rf_model.predict(X=[[0.0,0.08102822303771973,0.1749711036682129,0.3909306526184082,0.10950136184692383,0.09694719314575195,0.09722113609313965,0.07033276557922363],[0.0,0.4708571434020996,1.075922966003418,1.7906007766723633,0.1363675594329834,0.11570262908935547,0.13919687271118164,0.13847613334655762],[0.0,0.31091880798339844,0.4660007953643799,0.6013302803039551,0.10136079788208008,0.07022714614868164,0.08678197860717773,0.07829093933105469],[0.0,0.4898681640625,0.8029575347900391,1.273542881011963,0.1506366729736328,0.18001031875610352,0.14185094833374023,0.15266799926757812]])
for i in prediction:
    print(labels[i])

# Tunning Random Forest
# from itertools import product
# n_estimators = [50, 100, 25]
# criterion = ['gini', 'log_loss', 'entropy']
# max_features = [1, 'sqrt', 'log2']
# max_depths = [None, 2, 3, 4, 5]
# for n, f, d, c in product(n_estimators, max_features, max_depths, criterion): # with product we can iterate through all possible combinations
#     rf = RandomForestClassifier(n_estimators=n, criterion=c, max_features=f, max_depth=d, n_jobs=2, random_state=1337)
#     rf.fit(rf_train, train_label)
#     prediction_test = rf.predict(X=rf_test)
#     print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(test_label,prediction_test)))
#     # cm = confusion_matrix(test_label, prediction_test)
#     # cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
#     # plt.figure()
#     # plot_confusion_matrix(cm_norm, classes=rf.classes_, title='Confusion matrix accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(test_label,prediction_test)))

#drawing tree
# from sklearn import tree
# for tree_in_forest in rf_model.estimators_:
#     tree.plot_tree(tree_in_forest)
#     plt.show()
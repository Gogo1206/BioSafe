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
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=False)
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
    labels.append(filename[9:-4])
    data = []
    with open(filename, newline='') as csvfile:
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

# # confusion matrix
# cm = confusion_matrix(test_label, prediction_test)
# cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
# plot_confusion_matrix(cm_norm, classes=rf_model.classes_)

prediction = rf_model.predict(X=[[0,0,0,0,0,0,0,0]])
for i in prediction:
    print(labels[i])

# # Tunning Random Forest
# from itertools import product
# n_estimators = [50, 100, 25]
# criterion = ['gini', 'log_loss', 'entropy']
# # max_features = [1, 'sqrt', 'log2']
# accuracys = []
# max_depths = [2, 5, None]
# for n, d, c in product(n_estimators, max_depths, criterion): # with product we can iterate through all possible combinations
#     rf = RandomForestClassifier(n_estimators=n, criterion=c, max_depth=d)
#     rf.fit(rf_train, train_label)
#     prediction_test = rf.predict(X=rf_test)
#     accuracys.append(accuracy_score(test_label,prediction_test))
#     print("Model with n_estimator = {}\tcriterion = '{}'\tdepth = {}\taccuracy = {:.3f}".format(n, c, d, accuracys[-1]))
#     cm = confusion_matrix(test_label, prediction_test)
#     cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
#     # plt.figure()
#     # plot_confusion_matrix(cm_norm, classes=rf.classes_)
# print(accuracys)

# drawing tree
from sklearn import tree
for tree_in_forest in rf_model.estimators_:
    tree.plot_tree(tree_in_forest, fontsize=32, max_depth=1)
    plt.show()
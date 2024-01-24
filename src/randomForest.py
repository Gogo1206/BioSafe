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

# # confusion matrix
# cm = confusion_matrix(test_label, prediction_test)
# cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
# plot_confusion_matrix(cm_norm, classes=rf_model.classes_)


prediction = rf_model.predict(X=[[0.0,0.24199438095092773,0.4379723072052002,0.764326810836792,0.1516561508178711,0.14670944213867188,0.1560227870941162,0.11124801635742188],[0.0,0.05236196517944336,0.16290736198425293,0.39670515060424805,0.13989520072937012,0.11405229568481445,0.11712861061096191,0.06977033615112305],[0,0,0,0,0,0,0,0],[0.0,0.1505413055419922,0.5703155994415283,0.6908144950866699,0.08736705780029297,0.07372426986694336,0.07668042182922363,0.07169795036315918]])
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

# # drawing tree
# from sklearn import tree
# for tree_in_forest in rf_model.estimators_:
#     tree.plot_tree(tree_in_forest)
#     plt.show()
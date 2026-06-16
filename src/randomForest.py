import csv
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


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


def train_model():
    """Train Random Forest on keystroke timing data and evaluate."""
    files = glob.glob('tmp/data/*.csv')
    labels = []
    x = []
    y = []
    for filename in files:
        labels.append(filename[9:-4])
        with open(filename, newline='') as csvfile:
            reader = list(csv.reader(csvfile, delimiter=','))
            title = reader.pop(0)
            for row in reader:
                x.append([eval(row[i]) for i in range(len(row))])
                y.append(labels.index(filename[9:-4]))

    rf_train, rf_test, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=50)

    rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy')
    rf_model.fit(rf_train, train_label)

    prediction_test = rf_model.predict(X=rf_test)

    print("Training Accuracy is: ", rf_model.score(rf_train, train_label))
    print("Testing Accuracy is: ", rf_model.score(rf_test, test_label))

    prediction = rf_model.predict(X=[[0, 0, 0, 0, 0, 0, 0, 0]])
    for i in prediction:
        print(labels[i])

    # Draw first few decision trees
    from sklearn import tree
    for tree_in_forest in rf_model.estimators_:
        tree.plot_tree(tree_in_forest, fontsize=32, max_depth=1)
        plt.show()


if __name__ == "__main__":
    train_model()

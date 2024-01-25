from pynput import keyboard
import time
import csv
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random

start = time.time()
pressed = {}
press = [[]]
held = [[]]
password = []
counter = 0
def on_press(key): 
    if(key==keyboard.Key.esc):
        return False
    if key not in pressed: # Key was never pressed before
        pressed[key] = 0
    
    if pressed[key]==0: # Same logic1206
        if(len(password)!=4):
            password.append(key)
        pressed[key] = time.time()
        # print(key)
        # print('Key %s pressed at ' % key, time.time()) 
        press[len(press)-1].append(time.time()-start)
        if(len(press[-1])==4):
            global counter
            counter = counter+1
            press.append([])
            print("Count:", counter)

def on_release(key):  # Same logic
    # print('Key %s released at' % key, time.time())
    held.append([time.time()-pressed[key]]) if len(held[-1])==4 else held[-1].append(time.time()-pressed[key])
    pressed[key] = 0
    if(counter==10):
        return False

print("Please enter your 4 digit PIN for 10 times:")
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
listener.join()

with open("tmp\\run\\user.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])

    for i in range(len(press)):
        if(len(press[i])!=4):break
        row = []
        start = press[i][0]
        for j in press[i]:
            row.append(j-start)
        for j in held[i]:
            row.append(j)
        writer.writerow(row)
    # print(password)


files = glob.glob('tmp\\run\\*.csv')
labels = ["user"]
start = [[],[],[],[],[],[],[],[]]
end = [[],[],[],[],[],[],[],[]]
with open("tmp\\run\\others.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])
    for filename in files:
        if filename[8:-4] not in labels:continue
        x = [[],[],[],[],[],[],[],[]]
        data = []
        with open(filename, newline='') as csvfile:
            reader = list(csv.reader(csvfile, delimiter=','))
            title = reader.pop(0)
            # print(reader)
            for row in reader:
                for index in range(len(row)):
                    x[index].append(eval(row[index]))
                # plt.plot([1,2,3,4],[eval(row[i]) for i in range(4)], linestyle='dashed')
        q1 = np.percentile(x, 10, axis=1)
        q3 = np.percentile(x, 90, axis=1)
        # plt.plot([1,2,3,4],[q1[i] for i in range(4)], color='black')
        # plt.plot([1,2,3,4],[q3[i] for i in range(4)], color='black')
        # plt.show()

        for j in range(max(50,len(reader))):
            num1 = 0
            num2 = random.uniform(max(0,q1[1]-(q3[1]-q1[1])), q1[1]) if random.randint(0,1) == 1 else random.uniform(q3[1], q1[2])
            num3 = random.uniform(num2, q1[2]) if random.randint(0,1) == 1 else random.uniform(q3[2],q1[3])
            num4 = random.uniform(num3, q1[3]) if random.randint(0,1) == 1 else random.uniform(q3[3],q3[3]+(q3[3]-q3[1]))
            row = [num1, num2, num3, num4]
            # plt.plot([1,2,3,4],[q1[i] for i in range(4)], color='black')
            # plt.plot([1,2,3,4],[q3[i] for i in range(4)], color='black')
            # plt.plot([1,2,3,4],[i for i in row], linestyle='dashed')
            # plt.show()
            delays = [(random.uniform(0.05, q1[i]) if random.randint(0,1) == 1 else random.uniform(q3[i],0.20)) for i in range(4,8)]
            writer.writerow(row+delays)

with open("tmp\\run\\ai.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])

        for j in range(50):
            num = [random.randint(0,9) for i in range(4)]
            press = []
            held = []
            for digit in num:
                  start = time.time()
                  for guess in range(10):
                        if(guess==digit):
                            press.append(time.time())
                            held.append(time.time()-start)
                        # print(digit)
            row = []
            start = press[0]
            for j in press:
                row.append(j-start)
            for j in held:
                row.append(j)
            # plt.plot([1,2,3,4],[i for i in row[0:4]], linestyle='dashed')
            # plt.show()
            writer.writerow(row)

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

files = glob.glob('tmp\\run\\*.csv')
labels = []
x = []
y = []
for filename in files:
    labels.append(filename[8:-4])
    data = []
    with open(filename, newline='') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))
        title = reader.pop(0)
        # print(reader)
        for row in reader:
            # plt.title(filename)
            # plt.plot([1,2,3,4],[eval(row[i]) for i in range(4)], linestyle='dashed')
            x.append([eval(row[i]) for i in range(len(row))])
            y.append(labels.index(filename[8:-4]))
    # plt.show()

rf_train, rf_test, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=50)

rf_model = RandomForestClassifier(n_estimators=50, criterion='entropy')
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

x = [[]]
y = [[]]
def predict(key): 
    if(key==keyboard.Key.enter):
        return False
    if key not in pressed: # Key was never pressed before
        pressed[key] = 0
    
    if pressed[key]==0: # Same logic
        if(len(password)!=4):password.append(key)
        pressed[key] = time.time()
        # print('Key %s pressed at ' % key, time.time()) 
        x.append([time.time()-start]) if len(x[len(x)-1])==4 else x[-1].append(time.time()-start)


def on_release(key):  # Same logic
    # print('Key %s released at' % key, time.time())
    # held.append(time.time()-pressed[key])
    y.append([time.time()-pressed[key]]) if len(y[-1])==4 else y[-1].append(time.time()-pressed[key])
    pressed[key] = 0

print("Please enter your 4 digit PIN:")
listener = keyboard.Listener(on_press=predict, on_release=on_release)
listener.start()
listener.join()

for i in range(len(x)):
    if(len(x[i])!=4):break
    row = []
    start = x[i][0]
    for j in x[i]:
        row.append(j-start)
    for j in y[i]:
        row.append(j)
    prediction = rf_model.predict(X=[row])
    print(labels[prediction[0]])

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
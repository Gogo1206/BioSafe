from pynput import keyboard
import time
import csv
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
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
        for row in reader:
            x.append([eval(row[i]) for i in range(len(row))])
            y.append(labels.index(filename[9:-4]))

rf_train, rf_test, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=50)

print("Training Random Forest Model...")

rf_model = RandomForestClassifier(n_estimators=50, criterion='entropy')
rf_model.fit(rf_train, train_label)
prediction_test = rf_model.predict(X=rf_test)

print("Training Accuracy is: ", rf_model.score(rf_train, train_label))
print("Testing Accuracy is: ", rf_model.score(rf_test, test_label))


start = time.time()
pressed = {}
press = [[]]
held = [[]]
password = [1,2,0,6]
counter = 0
x = []
y = []
def enter_predict(x, y):
    row = []
    start = x[0]
    for i in x:
        row.append(i-start)
    for i in y:
        row.append(i)
    prediction = rf_model.predict(X=[row])
    print(labels[prediction[0]],"detected")

def on_press(key): 
    if(key==keyboard.Key.enter):
        print("AI Hacking")
        for i in range(50):
            print('\''+str(random.randint(0,9))+'\'', end=' ')
        print()
        for i in password:
            print(i,end='')
        print()
        enter_predict([0,0,0,0],[0,0,0,0])
    if key not in pressed:
        pressed[key] = 0
    
    if pressed[key]==0:
        pressed[key] = time.time()
        x.append(time.time()-start)


def on_release(key):
    y.append(time.time()-pressed[key])
    pressed[key] = 0
    if len(y)==4:
        enter_predict(x, y)
        x.clear()
        y.clear()

print("Try login with your 4 digit PIN:")
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
listener.join()
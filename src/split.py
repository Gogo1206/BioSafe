import csv
import matplotlib.pyplot as plt
import numpy as np

with open('tmp/train.csv', newline='') as csvfile:
    reader = list(csv.reader(csvfile, delimiter=','))
    reader.pop(0)
    data = []
    label = []
    for row in reader[0:8]:
        label.append(row[0])
        row.pop(0)
        for chunk in [row[x:x+4*2:2] for x in range(0, len(row), 4*2)]:
            if(len(chunk)!=4):continue
            group  = [eval(chunk[i]) for i in range(4)]
            held = []
            for i in range(1, len(group)):
                held.append(group[i]-group[i-1])
            standard = []
            for i in range(len(group)):
                standard.append((group[i]-group[0])/(group[3]-group[0]))
            mx = [0,0,0,0]
            mn = [1,1,1,1]
            plt.plot([1,2,3,4],[i for i in standard], linestyle='dashed')
            for i in range(len(standard)):
                mx[i] = max(mx[i], standard[i])
                mn[i] = min(mn[i], standard[i])
            data.append([i for i in standard])
    mean_x = np.average(data, axis=0)
    plt.plot([1,2,3,4],[i for i in mean_x], color='black')
plt.show()
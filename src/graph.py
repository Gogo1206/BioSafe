import csv
import matplotlib.pyplot as plt
import numpy as np

with open('press.csv', newline='') as csvfile:
    reader = list(csv.reader(csvfile, delimiter=','))
    reader.pop(0)
    print(reader)
    data = []
    for row in reader:
        plt.plot([1,2,3,4],[eval(i) for i in row], linestyle='dashed')
        data.append([eval(i) for i in row])
    mean_x = np.average(data, axis=0)
    plt.plot([1,2,3,4],[i for i in mean_x], color='black')
plt.show()
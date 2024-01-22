import csv
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random

files = glob.glob('tmp\\data\\*.csv')
labels = ["0308", "0816", "1206", "3221", "7958", "mom0227", "mom1206"]
start = [[],[],[],[],[],[],[],[]]
end = [[],[],[],[],[],[],[],[]]
for filename in files:
    if filename[9:-4] not in labels:continue
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
    with open("tmp/data/others.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])
        for j in range(50):
            num = random.uniform(0, q1[1]) if random.randint(0,1) == 1 else random.uniform(q3[1], q1[2])
            num2 = random.uniform(num, q1[2]) if random.randint(0,1) == 1 else random.uniform(q3[2],1)
            row = [0, num, num2, 1]
            # plt.plot([1,2,3,4],[q1[i] for i in range(4)], color='black')
            # plt.plot([1,2,3,4],[q3[i] for i in range(4)], color='black')
            # plt.plot([1,2,3,4],[i for i in row], linestyle='dashed')
            # plt.show()
            delays = [(random.uniform(0.05, q1[i]) if random.randint(0,1) == 1 else random.uniform(q3[i],0.20)) for i in range(4,8)]
            writer.writerow(row+delays)
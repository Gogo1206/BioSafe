import csv
import matplotlib.pyplot as plt
import numpy as np
import glob

# files = glob.glob('tmp\\data\\*.csv')
# labels = ["gogo"]
# for filename in files:
#     if filename[9:-4] not in labels:continue
#     with open(filename, newline='') as csvfile:
#         reader = list(csv.reader(csvfile, delimiter=','))
#         reader.pop(0)
#         # print(reader)
#         data = []
#         for row in reader:
#             plt.title(filename)
#             plt.plot([1,2,3,4],[eval(row[i]) for i in range(4)], linestyle='dashed')
#             data.append([eval(row[i]) for i in range(4)])
#         mean_x = np.average(data, axis=0)
#         plt.plot([1,2,3,4],[i for i in mean_x], color='black')
#     plt.show()

files = glob.glob('tmp\\data\\*.csv')
labels = ["user", "others","ai"]
for filename in files:
    if filename[9:-4] not in labels:continue
    with open(filename, newline='') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))
        reader.pop(0)
        # print(reader)
        data = []
        for row in reader:
            plt.title(filename)
            plt.plot([1,2,3,4,5,6,7,8],[eval(row[i]) for i in range(8)], linestyle='dashed')
            data.append([eval(row[i]) for i in range(8)])
        mean_x = np.average(data, axis=0)
        plt.plot([1,2,3,4,5,6,7,8],[i for i in mean_x], color='black')
    plt.show()
import csv
import matplotlib.pyplot as plt

with open('press.csv', newline='') as csvfile:
    reader = list(csv.reader(csvfile, delimiter=','))
    reader.pop(0)
    print(reader)
    for row in reader:
        plt.plot([1,2,3,4],[eval(i) for i in row])
plt.show()
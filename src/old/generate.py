import csv
import matplotlib.pyplot as plt
import numpy as np
import random


def generate_synthetic_data():
    """Generate synthetic keystroke data from real press data."""
    with open('tmp/press.csv', newline='') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))
        reader.pop(0)
        mx = [0, 0, 0, 0]
        mn = [1, 1, 1, 1]
        data = []
        for row in reader:
            for i in range(len(row)):
                mx[i] = max(mx[i], eval(row[i]))
                mn[i] = min(mn[i], eval(row[i]))
            data.append([eval(i) for i in row])
        mean_x = np.average(data, axis=0)

    with open("tmp/person.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4"])
        for i in range(50):
            row = []
            for j in range(4):
                row.append(random.uniform(mean_x[j] - max(mx[j] - mean_x[j], mean_x[j] - mn[j]),
                                         mean_x[j] + max(mx[j] - mean_x[j], mean_x[j] - mn[j])))
            writer.writerow(row)

    with open("tmp/others.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4"])
        for i in range(35):
            num = random.uniform(mean_x[1] - 5 * max(mx[1] - mean_x[1], mean_x[1] - mn[1]),
                                mean_x[1] - 1.1 * max(mx[1] - mean_x[1], mean_x[1] - mn[1])) if random.randint(0, 1) == 1 else random.uniform(mean_x[1] + 1.1 * max(mx[1] - mean_x[1], mean_x[1] - mn[1]), mean_x[1] + 5 * max(mx[1] - mean_x[1], mean_x[1] - mn[1]))
            num2 = random.uniform(max(num, mean_x[2] - 5 * max(mx[2] - mean_x[2], mean_x[2] - mn[2])),
                                 mean_x[2] - 1.1 * max(mx[2] - mean_x[2], mean_x[2] - mn[2])) if random.randint(0, 1) == 1 else random.uniform(mean_x[2] + 1.1 * max(mx[2] - mean_x[2], mean_x[2] - mn[2]), mean_x[2] + 5 * max(mx[2] - mean_x[2], mean_x[2] - mn[2]))
            row = [0, num, num2, 1]
            writer.writerow(row)
        for i in range(15):
            num = random.uniform(0, mean_x[1] - 1.5 * max(mx[1] - mean_x[1], mean_x[1] - mn[1])) if random.randint(0, 1) == 1 else random.uniform(mean_x[1] + 1.5 * max(mx[1] - mean_x[1], mean_x[1] - mn[1]), 1)
            row = [0, num, random.uniform(num, 1), 1]
            writer.writerow(row)

    with open('tmp/others.csv', newline='') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=','))
        reader.pop(0)
        data = []
        for row in reader:
            plt.plot([1, 2, 3, 4], [eval(i) for i in row], linestyle='dashed')
    plt.show()


if __name__ == "__main__":
    generate_synthetic_data()

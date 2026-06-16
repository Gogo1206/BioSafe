import csv
import numpy as np
import glob
import random


def generate_others_data():
    """Generate synthetic 'others' (impostor) keystroke data from user percentiles."""
    files = glob.glob('tmp/data/*.csv')
    labels = ["user"]
    with open("tmp/data/others.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])
        for filename in files:
            if filename[9:-4] not in labels:
                continue
            x = [[], [], [], [], [], [], [], []]
            with open(filename, newline='') as csvfile:
                reader = list(csv.reader(csvfile, delimiter=','))
                title = reader.pop(0)
                for row in reader:
                    for index in range(len(row)):
                        x[index].append(eval(row[index]))
            q1 = np.percentile(x, 15, axis=1)
            q3 = np.percentile(x, 85, axis=1)

            for j in range(max(50, len(reader))):
                num1 = 0
                num2 = random.uniform(max(0, q1[1] - (q3[1] - q1[1])), q1[1]) if random.randint(0, 1) == 1 else random.uniform(q3[1], q1[2])
                num3 = random.uniform(num2, q1[2]) if random.randint(0, 1) == 1 else random.uniform(q3[2], q1[3])
                num4 = random.uniform(num3, q1[3]) if random.randint(0, 1) == 1 else random.uniform(q3[3], q3[3] + (q3[3] - q1[3]))
                row = [num1, num2, num3, num4]
                delays = [(random.uniform(0.05, q1[i]) if random.randint(0, 1) == 1 else random.uniform(q3[i], 0.20)) for i in range(4, 8)]
                writer.writerow(row + delays)


if __name__ == "__main__":
    generate_others_data()

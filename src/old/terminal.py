from pynput import keyboard
import time
import csv
import numpy as np
import glob
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def run_full_pipeline():
    """Full pipeline: collect keystrokes, generate synthetic data, train, predict."""
    start = time.time()
    pressed = {}
    press = [[]]
    held = [[]]
    password = []
    counter = 0

    def on_press(key):
        if key == keyboard.Key.enter:
            return False
        if key not in pressed:
            pressed[key] = 0
        if pressed[key] == 0:
            if len(password) != 4:
                password.append(key)
            pressed[key] = time.time()
            print("*", end="")
            press[len(press) - 1].append(time.time() - start)
            if len(press[-1]) == 4:
                nonlocal counter
                counter = counter + 1
                press.append([])
                print("\nCount:", counter)

    def on_release(key):
        held.append([time.time() - pressed[key]]) if len(held[-1]) == 4 else held[-1].append(time.time() - pressed[key])
        pressed[key] = 0
        if counter == 10:
            return False

    print("Please enter your 4 digit PIN for 10 times:")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

    with open("tmp/run/correct user.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])
        for i in range(len(press)):
            if len(press[i]) != 4:
                break
            row = []
            start_time = press[i][0]
            for j in press[i]:
                row.append(j - start_time)
            for j in held[i]:
                row.append(j)
            writer.writerow(row)

    print("Analyzing Data...")
    files = glob.glob('tmp/run/*.csv')
    labels = ["correct user"]
    with open("tmp/run/others user.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])
        for filename in files:
            if filename[8:-4] not in labels:
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
                num4 = random.uniform(num3, q1[3]) if random.randint(0, 1) == 1 else random.uniform(q3[3], q3[3] + (q3[3] - q3[1]))
                row = [num1, num2, num3, num4]
                delays = [(random.uniform(0.05, q1[i]) if random.randint(0, 1) == 1 else random.uniform(q3[i], 0.20)) for i in range(4, 8)]
                writer.writerow(row + delays)

    with open("tmp/run/ai hacking.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])
        for j in range(50):
            num = [random.randint(0, 9) for i in range(4)]
            press_ai = []
            held_ai = []
            for digit in num:
                t_start = time.time()
                for guess in range(10):
                    if guess == digit:
                        press_ai.append(time.time())
                        held_ai.append(time.time() - t_start)
            row = []
            t_start = press_ai[0]
            for j in press_ai:
                row.append(j - t_start)
            for j in held_ai:
                row.append(j)
            writer.writerow(row)

    print("Training Random Forest Model...")
    files = glob.glob('tmp/run/*.csv')
    labels = []
    x = []
    y = []
    for filename in files:
        labels.append(filename[8:-4])
        with open(filename, newline='') as csvfile:
            reader = list(csv.reader(csvfile, delimiter=','))
            title = reader.pop(0)
            for row in reader:
                x.append([eval(row[i]) for i in range(len(row))])
                y.append(labels.index(filename[8:-4]))

    rf_train, rf_test, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=50)

    rf_model = RandomForestClassifier(n_estimators=50, criterion='entropy')
    rf_model.fit(rf_train, train_label)
    prediction_test = rf_model.predict(X=rf_test)

    print("Training Accuracy is: ", rf_model.score(rf_train, train_label))
    print("Testing Accuracy is: ", rf_model.score(rf_test, test_label))

    x_vals = []
    y_vals = []

    def enter_predict(x_vals, y_vals):
        row = []
        start_t = x_vals[0]
        for i in x_vals:
            row.append(i - start_t)
        for i in y_vals:
            row.append(i)
        prediction = rf_model.predict(X=[row])
        print(labels[prediction[0]], "detected")

    def on_press_predict(key):
        if key == keyboard.Key.enter:
            print("AI Hacking")
            for i in range(50):
                print('\'' + str(random.randint(0, 9)) + '\'', end=' ')
            print()
            for i in password:
                print(i, end='')
            print()
            enter_predict([0, 0, 0, 0], [0, 0, 0, 0])
        if key not in pressed:
            pressed[key] = 0
        if pressed[key] == 0:
            pressed[key] = time.time()
            x_vals.append(time.time() - start)

    def on_release_predict(key):
        y_vals.append(time.time() - pressed[key])
        pressed[key] = 0
        if len(y_vals) == 4:
            enter_predict(x_vals, y_vals)
            x_vals.clear()
            y_vals.clear()

    print("Try login with your 4 digit PIN:")
    listener = keyboard.Listener(on_press=on_press_predict, on_release=on_release_predict)
    listener.start()
    listener.join()


if __name__ == "__main__":
    run_full_pipeline()

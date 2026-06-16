from pynput import keyboard
import time
import csv
import numpy as np
import glob
import random
from tkinter import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def run(start, pressed, press, held):
    count_label.grid_remove()
    hint_label.configure(text="Processing...")
    with open("tmp/run/user.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])

        for i in range(len(press)):
            if len(press[i]) != 4:
                break
            row = []
            start_t = press[i][0]
            for j in press[i]:
                row.append(j - start_t)
            for j in held[i]:
                row.append(j)
            writer.writerow(row)

    hint_label.configure(text="Protecting You From Other Users...")
    files = glob.glob('tmp/run/*.csv')
    labels = ["user"]
    with open("tmp/run/others.csv", 'w', newline='') as csvfile:
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
            q1 = np.percentile(x, 10, axis=1)
            q3 = np.percentile(x, 90, axis=1)

            for j in range(max(50, len(reader))):
                num1 = 0
                num2 = random.uniform(max(0, q1[1] - (q3[1] - q1[1])), q1[1]) if random.randint(0, 1) == 1 else random.uniform(q3[1], q1[2])
                num3 = random.uniform(num2, q1[2]) if random.randint(0, 1) == 1 else random.uniform(q3[2], q1[3])
                num4 = random.uniform(num3, q1[3]) if random.randint(0, 1) == 1 else random.uniform(q3[3], q3[3] + (q3[3] - q3[1]))
                row = [num1, num2, num3, num4]
                delays = [(random.uniform(0.05, q1[i]) if random.randint(0, 1) == 1 else random.uniform(q3[i], 0.20)) for i in range(4, 8)]
                writer.writerow(row + delays)

    hint_label.configure(text="Protecting You From AI...")
    with open("tmp/run/ai.csv", 'w', newline='') as csvfile:
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

    hint_label.configure(text="Training Model...")
    files = glob.glob('tmp/run/*.csv')
    labels_list = []
    x = []
    y = []
    for filename in files:
        labels_list.append(filename[8:-4])
        with open(filename, newline='') as csvfile:
            reader = list(csv.reader(csvfile, delimiter=','))
            title = reader.pop(0)
            for row in reader:
                x.append([eval(row[i]) for i in range(len(row))])
                y.append(labels_list.index(filename[8:-4]))

    rf_train, rf_test, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=50)

    rf_model = RandomForestClassifier(n_estimators=50, criterion='entropy')
    rf_model.fit(rf_train, train_label)
    prediction_test = rf_model.predict(X=rf_test)

    hint_label.configure(text="Training Accuracy is: " + str(rf_model.score(rf_train, train_label)) +
                         "\nTesting Accuracy is: " + str(rf_model.score(rf_test, test_label)))


def main():
    global counter, start, pressed, press, held, window, pass_var, hint_label, entry, count_label, listener

    counter = 0
    start = time.time()
    pressed = {}
    press = [[]]
    held = [[]]

    window = Tk()
    pass_var = StringVar()
    hint_label = Label(window, text="Please Enter Your Password:", font=('calibre', 20, 'bold'))
    entry = Entry(window, textvariable=pass_var, font=('calibre', 20, 'normal'), show='*')
    count_label = Label(window, text="", font=('calibre', 20, 'normal'))
    hint_label.grid(row=0, column=0)
    entry.grid(row=1, column=0)
    count_label.grid(row=2, column=0)

    def on_press(key):
        if key == keyboard.Key.esc:
            listener.stop()
            run(start, pressed, press, held)
        if key not in pressed:
            pressed[key] = 0
        if pressed[key] == 0:
            pressed[key] = 1
            press[-1].append(time.time() - start)
            if len(press[-1]) == 4:
                press.append([])

    def on_release(key):
        pressed[key] = 0
        held[-1].append(time.time() - pressed[key])
        if len(held[-1]) == 4:
            held.append([])
            global counter
            counter = counter + 1
            pass_var.set("")
            count_label.configure(text="Counter:" + str(counter))
        if counter == 5:
            listener.stop()
            run(start, pressed, press, held)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    window.mainloop()


if __name__ == "__main__":
    main()

from pynput import keyboard
import time
import csv
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def run_terminal_demo():
    """Terminal-based keystroke capture with real-time Random Forest prediction."""
    files = glob.glob('tmp/data/*.csv')
    labels = []
    x = []
    y = []
    for filename in files:
        labels.append(filename[9:-4])
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
    password = [1, 2, 0, 6]
    counter = 0
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

    def on_press(key):
        if key == keyboard.Key.enter:
            print("AI Hacking")
            display = "****"
            guess = 0
            for j in range(4):
                for i in range(0, 9):
                    guess = (int)(guess / 10) * 10 + i
                    print((str)(guess) + display[j + 1:])
                    if i == password[j]:
                        guess *= 10
                        break
            enter_predict([0, 0, 0, 0], [0, 0, 0, 0])
        if key not in pressed:
            pressed[key] = 0
        if pressed[key] == 0:
            pressed[key] = time.time()
            x_vals.append(time.time() - start)

    def on_release(key):
        y_vals.append(time.time() - pressed[key])
        pressed[key] = 0
        if len(y_vals) == 4:
            enter_predict(x_vals, y_vals)
            x_vals.clear()
            y_vals.clear()

    print("Try login with your 4 digit PIN:")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()


if __name__ == "__main__":
    run_terminal_demo()

from tkinter import *
from pynput import keyboard
import time
import csv
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def run_gui():
    """Tkinter GUI login with keystroke-based identity verification."""
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

    window = Tk()
    window.geometry("600x400")
    window.grid_rowconfigure(0, weight=1)
    window.grid_rowconfigure(4, weight=1)
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=2)
    window.grid_columnconfigure(2, weight=1)
    pass_var = StringVar()
    hint_label = Label(window, text="Please Enter Your Password:", font=('calibre', 20, 'bold'))
    entry = Entry(window, textvariable=pass_var, font=('calibre', 20, 'normal'), show='*')
    text_label = Label(window, text="Model accuracy of {:.3f}".format(rf_model.score(rf_test, test_label)), font=('calibre', 20, 'normal'))
    empty = Label(window, text="")
    hint_label.grid(row=1, column=1)
    entry.grid(row=2, column=1)
    text_label.grid(row=3, column=1)
    empty.grid(row=4, column=2)

    start = time.time()
    pressed = {}
    password = [1, 2, 0, 6]
    x_vals = []
    y_vals = []

    def enter_predict(x_vals, y_vals):
        row = []
        start_t = x_vals[0]
        for i in x_vals:
            row.append(i - start_t)
        for i in y_vals:
            row.append(i)
        if entry.get() == "1206":
            prediction = rf_model.predict(X=[row])
            text_label.config(text=labels[prediction[0]] + " detected")
        else:
            text_label.config(text="wrong password")
        entry.delete(0, END)

    def on_press(key):
        if len(x_vals) == 4:
            return
        if key == keyboard.Key.enter:
            text_label.config(text='ai hacking')
            entry.config(show='')
            for i in range(4):
                for n in range(0, 9):
                    entry.insert(i, n)
                    time.sleep(0.1)
                    if n == password[i]:
                        break
                    entry.delete(i, i + 1)
            enter_predict([0, 0, 0, 0], [0, 0, 0, 0])
            entry.config(show='*')
            return
        if key == keyboard.Key.backspace:
            x_vals.clear()
            entry.delete(0, END)
            return
        if key not in pressed:
            pressed[key] = 0
        if pressed[key] == 0:
            pressed[key] = time.time()
            x_vals.append(time.time() - start)

    def on_release(key):
        if len(y_vals) == 4:
            return
        if key == keyboard.Key.enter:
            return
        if key == keyboard.Key.backspace:
            y_vals.clear()
            entry.delete(0, END)
            return
        y_vals.append(time.time() - pressed[key])
        pressed[key] = 0
        if len(y_vals) == 4:
            enter_predict(x_vals, y_vals)
            x_vals.clear()
            y_vals.clear()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    window.mainloop()


if __name__ == "__main__":
    run_gui()

from pynput import keyboard
import time
import csv


def collect_keystrokes():
    """Capture 4-digit PIN keystroke timing and write to CSV."""
    start = time.time()
    pressed = {}
    press = [[]]
    held = [[]]

    def on_press(key):
        if key == keyboard.Key.esc:
            return False
        if key not in pressed:
            pressed[key] = 0
        if pressed[key] == 0:
            pressed[key] = time.time()
            print('Key %s pressed at ' % key, time.time())
            press.append([time.time() - start]) if len(press[len(press) - 1]) == 4 else press[len(press) - 1].append(time.time() - start)

    def on_release(key):
        print('Key %s released at' % key, time.time())
        held.append([time.time() - pressed[key]]) if len(held[-1]) == 4 else held[-1].append(time.time() - pressed[key])
        pressed[key] = 0

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

    with open("tmp/press.csv", 'w', newline='') as csvfile:
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


if __name__ == "__main__":
    collect_keystrokes()

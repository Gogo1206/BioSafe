from pynput import keyboard
import time
import csv


start = time.time()
pressed = {}
press = [[]]
held = []
def on_press(key): 
    if(key==keyboard.Key.esc):
        return False
    if key not in pressed: # Key was never pressed before
        pressed[key] = 0
    
    if pressed[key]==0: # Same logic
        pressed[key] = time.time()
        print('Key %s pressed at ' % key, time.time()) 
        press.append([time.time()-start]) if len(press[len(press)-1])==4 else press[len(press)-1].append(time.time()-start)

def on_release(key):  # Same logic
    print('Key %s released at' % key, time.time())
    held.append(time.time()-pressed[key])
    pressed[key] = 0


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()
listener.join()



with open("press.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["press1", "press2", "press3", "press4"])
    for i in press:
        row = []
        for j in reversed(i):
            row.append((j-i[0])/(i[3]-i[0]))
        writer.writerow(reversed(row))
from pynput import keyboard
import time
import numpy as np

start = time.time()
pressed = {}
press = [[]]
held = []
password = []
def on_press(key): 
    if(key==keyboard.Key.esc):
        return False
    if key not in pressed: # Key was never pressed before
        pressed[key] = 0
    
    if pressed[key]==0: # Same logic
        if(len(password)!=4):password.append(key)#
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

for i in range(len(press)):
    row = [np.average(held[i])]
    for j in reversed(press[i]):
        row.append((j-press[i][0])/(press[i][3]-press[i][0]))
print(rf_model.predict(X=reversed(row)))
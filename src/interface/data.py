from pynput import keyboard
import time
import csv
import numpy as np

start = time.time()
pressed = {}
press = [[]]
held = [[]]
password = []
def on_press(key):
    if(key==keyboard.Key.esc):
        return False
    if key not in pressed: # Key was never pressed before
        pressed[key] = 0

    if pressed[key]==0: # Same logic
        if(len(password)!=4):password.append(key-3)
        pressed[key] = time.time()
        print('Key %s pressed at ' % key, time.time()) 
        press.append([time.time()-start]) if len(press[len(press)-1])==4 else press[len(press)-1].append(time.time()-start)

def on_release(key):  # Same logic
    print('Key %s released at' % key, time.time())
    # held.append(time.time()-pressed[key])
    held.append([time.time()-pressed[key]]) if len(held[-1])==4 else held[-1].append(time.time()-pressed[key])
    pressed[key] = 0


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
listener.join()



with open("tmp/inerface/user.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])

    #normalize
    # for i in range(len(press)):
    #     if(len(press[i])!=4):break
    #     row = []
    #     for j in reversed(held[i]):
    #         row.append(j)
    #     for j in reversed(press[i]):
    #         row.append((j-press[i][0])/(press[i][3]-press[i][0]))
    #     writer.writerow(reversed(row))

    #not normalized08050805
    for i in range(len(press)):
        if(len(press[i])!=4):break
        row = []
        start = press[i][0]
        for j in press[i]:
            row.append(j-start)
        for j in held[i]:
            row.append(j)
        writer.writerow(row)
    print(password[0])

import csv
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random

files = glob.glob('tmp\\data\\*.csv')
labels = ["user"]
start = [[],[],[],[],[],[],[],[]]
end = [[],[],[],[],[],[],[],[]]
with open("tmp/interface/others.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])
    for filename in files:
        if filename[14:-4] not in labels:continue
        x = [[],[],[],[],[],[],[],[]]
        data = []
        with open(filename, newline='') as csvfile:
            reader = list(csv.reader(csvfile, delimiter=','))
            title = reader.pop(0)
            # print(reader)
            for row in reader:
                for index in range(len(row)):
                    x[index].append(eval(row[index]))
                # plt.plot([1,2,3,4],[eval(row[i]) for i in range(4)], linestyle='dashed')
        q1 = np.percentile(x, 15, axis=1)
        q3 = np.percentile(x, 85, axis=1)
        # plt.plot([1,2,3,4],[q1[i] for i in range(4)], color='black')
        # plt.plot([1,2,3,4],[q3[i] for i in range(4)], color='black')
        # plt.show()

        #normalized
        # for j in range(100):
        #     # num = random.uniform(0, q1[1]) if random.randint(0,1) == 1 else random.uniform(q3[1], q1[2])
        #     # num2 = random.uniform(num, q1[2]) if random.randint(0,1) == 1 else random.uniform(q3[2],1)
        #     num = random.uniform(max(0,q1[1]-(q3[1]-q1[1])), q1[1]) if random.randint(0,1) == 1 else random.uniform(q3[1], q1[2])
        #     num2 = random.uniform(num, q1[2]) if random.randint(0,1) == 1 else random.uniform(q3[2],q3[2]+(q3[2]-q1[2]))
        #     row = [0, num, num2, 1]
        #     # plt.plot([1,2,3,4],[q1[i] for i in range(4)], color='black')
        #     # plt.plot([1,2,3,4],[q3[i] for i in range(4)], color='black')
        #     # plt.plot([1,2,3,4],[i for i in row], linestyle='dashed')
        #     # plt.show()
        #     delays = [(random.uniform(0.05, q1[i]) if random.randint(0,1) == 1 else random.uniform(q3[i],0.20)) for i in range(4,8)]
        #     writer.writerow(row+delays)

        #not normalized
        for j in range(max(50,len(reader))):
            num1 = 0
            num2 = random.uniform(max(0,q1[1]-(q3[1]-q1[1])), q1[1]) if random.randint(0,1) == 1 else random.uniform(q3[1], q1[2])
            num3 = random.uniform(num2, q1[2]) if random.randint(0,1) == 1 else random.uniform(q3[2],q1[3])
            num4 = random.uniform(num3, q1[3]) if random.randint(0,1) == 1 else random.uniform(q3[3],q3[3]+(q3[3]-q3[1]))
            row = [num1, num2, num3, num4]
            # plt.plot([1,2,3,4],[q1[i] for i in range(4)], color='black')
            # plt.plot([1,2,3,4],[q3[i] for i in range(4)], color='black')
            # plt.plot([1,2,3,4],[i for i in row], linestyle='dashed')
            # plt.show()
            delays = [(random.uniform(0.05, q1[i]) if random.randint(0,1) == 1 else random.uniform(q3[i],0.20)) for i in range(4,8)]
            writer.writerow(row+delays)

import random
import csv
import time
import matplotlib.pyplot as plt

with open("tmp/interface/ai.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])

        for j in range(50):
            num = [random.randint(0,9) for i in range(4)]
            press = []
            held = []
            for digit in num:
                  start = time.time()
                  for guess in range(10):
                        if(guess==digit):
                            press.append(time.time())
                            held.append(time.time()-start)
                        # print(digit)
            row = []
            start = press[0]
            for j in press:
                row.append(j-start)
            for j in held:
                row.append(j)
            # plt.plot([1,2,3,4],[i for i in row[0:4]], linestyle='dashed')
            # plt.show()
            writer.writerow(row)
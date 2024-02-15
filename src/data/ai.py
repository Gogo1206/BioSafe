import random
import csv
import time
import matplotlib.pyplot as plt

with open("tmp/data/ai.csv", 'w', newline='') as csvfile:
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
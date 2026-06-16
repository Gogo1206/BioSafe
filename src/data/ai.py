import random
import csv
import time


def generate_ai_data():
    """Generate synthetic AI brute-force keystroke timing data."""
    with open("tmp/data/ai.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["press1", "press2", "press3", "press4", "delay1", "delay2", "delay3", "delay4"])

        for j in range(50):
            num = [random.randint(0, 9) for i in range(4)]
            press = []
            held = []
            for digit in num:
                start = time.time()
                for guess in range(10):
                    if guess == digit:
                        press.append(time.time())
                        held.append(time.time() - start)
            row = []
            start = press[0]
            for j in press:
                row.append(j - start)
            for j in held:
                row.append(j)
            writer.writerow(row)


if __name__ == "__main__":
    generate_ai_data()

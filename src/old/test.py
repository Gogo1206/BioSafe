from tkinter import *


def main():
    win = Tk()
    win.geometry("750x250")
    win.grid_rowconfigure(0, weight=1)
    win.grid_columnconfigure(0, weight=1)
    label = Label(win, text="This is a Centered Text", font=('Aerial 15 bold'))
    label.grid(row=1, column=0)
    label.grid_rowconfigure(1, weight=1)
    label.grid_columnconfigure(1, weight=1)
    win.mainloop()


if __name__ == "__main__":
    main()

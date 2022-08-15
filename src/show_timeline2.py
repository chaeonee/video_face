# -*- coding: utf-8 -*-
from PIL import Image, ImageTk

import numpy as np

import tkinter as tk
import os
file_name = ''
def makeTimeline1():
    os.system(r'"../src/scrolltest1.py"')

def makeTimeline2():
    os.system(r'"../src/scrolltest2.py"')

def makeTimeline3():
    os.system(r'"../src/scrolltest3.py"')
if __name__ == '__main__':
   # makeTimeline("vtest4.mp4")

    root_ = tk.Tk()
    root_.title("Main Character=")

    button = tk.Button(root_, text="test1", command=lambda: makeTimeline1(),overrelief="solid", width=15, repeatdelay=1000, repeatinterval=100)
    button2 = tk.Button(root_, text="test2", command=lambda: makeTimeline2(),overrelief="solid", width=15, repeatdelay=1000, repeatinterval=100)
    button3 = tk.Button(root_, text="test3", command=lambda: makeTimeline3(),overrelief="solid", width=15, repeatdelay=1000, repeatinterval=100)
    button.pack()
    button2.pack()
    button3.pack()
    root_.mainloop()

import numpy as np
from PIL import Image
import pickle
import os.path
import math
import random
import re
import common
import random_points
from tkinter import *
from PIL import Image, ImageTk
from collections import defaultdict

lasfile = 'E:\\workspaces\\LIDAR_WORKSPACE\\lidar\\kita.txt'


imagewidth, imageheight = 4000, 4000

class Timelapse:
    def __init__(self, lasfile):
        self.lasfile = lasfile
        self.curridx = 0
        self.X = np.zeros((imagewidth, imageheight))
        self.load_lidar()
        self.init_tkinter()
        pass

    def load_lidar(self):
        # load the las file
        lines = open(self.lasfile, 'r').readlines()
        points = []
        for line in lines:
            parts = line.split(' ')
            x = float(parts[0])
            y = float(parts[1])
            a = float(parts[3])
            points.append((x, y, a))

        # find mins
        minx, miny = 10000000, 10000000
        for i in range(len(points)):
            if points[i][0] < minx:
                minx = points[i][0]
            if points[i][1] < miny:
                miny = points[i][1]

        # normalize points
        for i in range(len(points)):
            points[i] = (points[i][0] - minx, points[i][1] - miny, points[i][2])
        self.points = points

    def render_image_on_canvas(self, data):
        self.im = Image.frombytes('L', (data.shape[1], data.shape[0]), data.astype('b').tostring())
        self.photo = ImageTk.PhotoImage(image=self.im)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.currimage = data


    def init_tkinter(self):
        self.root = Tk()
        self.frame = Frame(self.root, width=imagewidth, height=imageheight)
        self.frame.pack()

        # canvas
        self.canvas = Canvas(self.frame, width=imagewidth, height=imageheight, scrollregion=(0, 0, imagewidth, imageheight))
        hbar = Scrollbar(self.frame, orient=HORIZONTAL)
        hbar.pack(side=BOTTOM, fill=X)
        hbar.config(command=self.canvas.xview)
        vbar = Scrollbar(self.frame, orient=VERTICAL)
        vbar.pack(side=RIGHT, fill=Y)
        vbar.config(command=self.canvas.yview)
        self.canvas.config(width=imagewidth, height=imageheight)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        self.canvas.place(x=-2, y=-2)

        self.canvas.pack(side=LEFT, expand=True, fill=BOTH)

        self.root.update()
        self.update_clock()
        self.root.mainloop()

    def update_clock(self):
        for i in range(10000):
            x = (float(imagewidth) / 1000.0) * self.points[self.curridx][0]
            y = (float(imageheight) / 1000.0) * self.points[self.curridx][1]
            a = self.points[self.curridx][2]
            rgb = (0,0,0)
            if a >= -30 and a < -10:
                rgb = (int((a + 30) * 4), 0, 0)
            if a >= -10 and a < 10:
                rgb = (0, int((a + 30) * 4), 0)
            if a >= 10 and a <= 30:
                rgb = (0, 0, int((a + 30) * 4))
            rgb = "#%02x%02x%02x" % rgb
            self.curridx += 5
            self.canvas.create_oval(max(x - 3, 0), max(y - 3, 0), min(x + 3, imageheight), min(y + 3, imageheight), fill=rgb)
        self.root.after(1, self.update_clock)



a = Timelapse(lasfile)
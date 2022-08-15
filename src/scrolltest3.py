# -*- coding: utf-8 -*-
import tkinter as tk

from tkinter import *
from PIL import Image, ImageTk
width_ = 1150

def findRelationship(file):
    f = open("../data/"+file+"/"+file+".txt", 'r')
    lines = f.readlines()

    count_array = []
    for i in range(4, len(lines)):
        for j in range(i+1, len(lines)):
            l1 = lines[i].split()
            l2 = lines[j].split()

            count = 0
            for k in range(len(l1)):
                tmp1 = l1[k].split('_')
                tmp1 = tmp1[0]
                for l in range(len(l2)):
                    tmp2 = l2[l].split('_')
                    tmp2 = tmp2[0]
                    if (tmp1 == tmp2):
                        count_array.append([l1[1], l2[1], l1[k], count])
                        count = count + 1

    f.close()
    return count_array

def data(canvas,frame):
    #root = tk.Tk()
    file = "vtest24.mp4"
    ##
    f = open("../data/"+file+"/"+file+".txt", 'r')


    tmp = f.readline()

    runningTime = float(tmp[:-1])


    frame = f.readline()
    frameNum = frame.split()


    lastFrame = frameNum[len(frameNum)-1]


    character_count = f.readline()
    character_count = character_count.splitlines()



     #draw Timebar
    canvas.create_line(255,50, width_, 50, width=5 )    # total timebar
    canvas.create_line(255,40, 255, 60, width=5 )       # start timebar
    canvas.create_line(width_,40, width_, 60, width=5)  # end timebar
    canvas.create_text(225, 50, text="00:00")

    print("running")
    print(runningTime)
    timebar =  int(runningTime) / 300
    timebar = int(timebar)

    print("timebar")
    print(timebar)

    ######
    timeleft = int(runningTime) % 300
    print("timeleft")
    print(timeleft)

    timeSpace = (width_-255)  / (timebar + timeleft/300)

    min_n = 5
    minu = "0"+str(min_n)

    hour = 0
    houru = "0"+str(hour)

    for i in range(1, timebar+1):
        canvas.create_line(255+timeSpace*i,40, 255+timeSpace*i, 60, width=2)
        if (min_n == 0 or min_n == 5):
            canvas.create_text(255+timeSpace*i, 30, text=houru+":"+"0"+str(min_n))
        else :
            canvas.create_text(255+timeSpace*i, 30, text=houru+":"+str(min_n))
        min_n = min_n + 5
        if ( min_n == 60 ):
            min_n = 0
            hour = hour + 1
            houru = "0" + str(hour)

    if ((int((int(runningTime)%3600)/60)) < 10):
        canvas.create_text(width_+30, 50, text="0"+str(int(runningTime/3600))+":0"+str(int((int(runningTime)%3600)/60)))
    else :
        canvas.create_text(width_+30, 50, text="0"+str(int(runningTime/3600))+":"+str(int((int(runningTime)%3600)/60)))
    '''
    for i in range(1, timebar+1):
        print("line_num")
        print(i)
        canvas.create_line(255+timeSpace*i,40, 255+timeSpace*i, 60, width=2)

        if ( i % 2 == 0):
            canvas.create_text(255+timeSpace*i, 30, text=minu+":"+"00")
        else :
            canvas.create_text(255+timeSpace*i, 30, text=minu+":"+"30")
            min_n = min_n+1
            minu = "0"+str(min_n)


    canvas.create_text(width_+30, 50, text=str(int(runningTime/60))+":"+str(int(runningTime)%60))
    '''



    lines = f.readlines()
    count = 0
    im = []
    for i in lines:
        character_emerge = i.split()
        if((len(character_emerge)-1)>=(float)(character_count[0])):
            for i in range(0,len(character_emerge)):
                for j in range(0,len(frameNum)-1):
                    tmp = character_emerge[i].split('_')
                    if(str(frameNum[j])==str(tmp[0])):
                        width__ = width_ - 255
                        check_start = (int)(frameNum[j])
                        check_end = (int)(frameNum[j+1])
                        ratio_start = int(float((width__/(int)(lastFrame))*check_start))
                        ratio_end = int(float((width__/(int)(lastFrame))*check_end))
                        thumbnail = Image.open("../data/"+file+"/"+character_emerge[1]).resize((100,100), Image.ANTIALIAS)
                        im.append(ImageTk.PhotoImage(thumbnail))
                        canvas.create_image(50, count*110+80, image=im[len(im)-1], anchor='nw')
                        canvas.create_rectangle(260+ratio_start, count*110+80, 250+ratio_end, 100+count*(110)+80, fill = 'red')
            count = count + 1

    f.close()




    list_count = findRelationship(file)

    f = open("../data/"+file+"/"+file+".txt", 'r')
    lines = f.readlines()

    char_line = lines[1].split()

    for i in range(len(list_count)):
        if (list_count[i][3] == 0):
            count = count + 1
            thumbnail1 = Image.open("../data/"+file+"/"+list_count[i][0]).resize((100,100), Image.ANTIALIAS)
            thumbnail2 = Image.open("../data/"+file+"/"+list_count[i][1]).resize((100,100), Image.ANTIALIAS)
            im.append(ImageTk.PhotoImage(thumbnail1))
            im.append(ImageTk.PhotoImage(thumbnail2))
            canvas.create_image(0, count*110+80, image=im[len(im)-2], anchor='nw')
            canvas.create_image(100, count*110+80, image=im[len(im)-1], anchor='nw')

            #타임바그리기
        for j in range(len(char_line)-1):
            tmp = list_count[i][2].split('_')
            if (str(char_line[j]) == str(tmp[0])):
                width__ = width_ - 255
                check_start = (int)(frameNum[j])
                check_end = (int)(frameNum[j+1])
                ratio_start = int(float((width__/(int)(lastFrame))*check_start))
                ratio_end = int(float((width__/(int)(lastFrame))*check_end))
                canvas.create_rectangle(260+ratio_start, count*110+80, 250+ratio_end, 100+count*(110)+80, fill = 'blue')



    canvas.pack()
    f.close()
    root.mainloop()


def myfunction(event):
    canvas.configure(scrollregion=canvas.bbox("all"),width=1150,height=700)

root=tk.Tk()
sizex = 1200
sizey = 2000
width = 500
height = 1000
posx  = 100
posy  = 100
root.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, posx, posy))

myframe=tk.Frame(root,relief=tk.GROOVE,width=500,height=1000,bd=1)
myframe.place(x=10,y=10)
canvas=tk.Canvas(myframe)
frame=tk.Frame(canvas)
myscrollbar=tk.Scrollbar(myframe,orient="vertical",command=canvas.yview)
myscrollbar2 = tk.Scrollbar(myframe,orient='horizontal',command=canvas.xview)
canvas.configure(yscrollcommand=myscrollbar.set)
canvas.configure(xscrollcommand=myscrollbar2.set)
myscrollbar.pack(side="right",fill="y")
myscrollbar2.pack(side="bottom",fill='x')
canvas.pack(side="left")
canvas.create_window((0,0),window=frame,anchor='nw')
myframe.bind("<Configure>",myfunction)
data(canvas,frame)
root.mainloop()

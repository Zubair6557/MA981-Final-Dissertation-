import csv
import datetime
import glob
import os
import shutil
import time
import tkinter as tk
import tkinter.font as font
import tkinter.ttk as ttk
from cProfile import label
from email.mime import image
from fileinput import filename
from importlib.resources import path
from tkinter import (BOTH, Canvas, Frame, Label, Message, PhotoImage, Text,
                     dialog, messagebox)
from tkinter.filedialog import Directory

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

Study = tk.Tk()
Study.title("face recogniser")
Study.geometry('1300x800')
img = ImageTk.PhotoImage(Image.open("Images and Screenshot/AI.jpeg"))
lab = Label(image=img)
lab.pack()

dialog_title = 'Quit'
dialog_text = 'Are You Sure?'


Study.grid_rowconfigure(0, weight=1)
Study.grid_columnconfigure(0, weight=1)

x_cord = 75;
y_cord = 20;
checker=0;

message = tk.Label(Study, text="UNIVERSITY OF ESSEX", bg="white"  ,fg="black"  ,width=30  ,height=2,font=('Times New Roman', 25, 'bold underline')) 
message.place(x=100, y=30)

message = tk.Label(Study, text="ATTENDANCE MANAGEMENT PORTAL", bg="white"   ,fg="black"  ,width=40  ,height=2,font=('Times New Roman', 25, 'bold underline')) 
message.place(x=650, y=30)

lbl = tk.Label(Study, text="Enter Your College ID",width=19  ,height=2  ,fg="black"  ,bg="white" ,font=('Times New Roman', 25, ' bold ') ) 
lbl.place(x=200-x_cord, y=200-y_cord)


txt = tk.Entry(Study,width=30,bg="white" ,fg="black",font=('Times New Roman', 15, ' bold '))
txt.place(x=200-x_cord, y=330-y_cord)

lbl2 = tk.Label(Study, text="Enter Your Name",width=19  ,fg="black"  ,bg="white"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
lbl2.place(x=600-x_cord, y=200-y_cord)

txt2 = tk.Entry(Study,width=30  ,bg="white"  ,fg="black",font=('Times New Roman', 15, ' bold ')  )
txt2.place(x=600-x_cord, y=330-y_cord)

lbl3 = tk.Label(Study, text="NOTIFICATION",width=24  ,fg="black"  ,bg="white"  ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
lbl3.place(x=1020-x_cord, y=200-y_cord)

message = tk.Label(Study, text="" ,bg="white"  ,fg="black"  ,width=39  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
message.place(x=1020-x_cord, y=330-y_cord)

lbl3 = tk.Label(Study, text="ATTENDANCE",width=20  ,fg="black"  ,bg="white"  ,height=2 ,font=('Times New Roman', 30, ' bold ')) 
lbl3.place(x=120, y=570-y_cord)


message2 = tk.Label(Study, text="" ,fg="black"   ,bg="lightblue",activeforeground = "green",width=60  ,height=4  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=570-y_cord)



def clear1():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


def TakeImages():  
    isheaderrow = 0
    Id=(txt.get())
    name=(txt2.get())
    if not Id:
        res="Please enter Id"
        message.configure(text = res)
        MsgBox = tk.messagebox.askquestion ("Warning","Please Enter Correct Enrollment Number",icon = 'warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Your need','Please go through the readme file properly')
    elif not name:
        res="Please enter Name"
        message.configure(text = res)
        MsgBox = tk.messagebox.askquestion ("Warning","Please enter your name properly , press yes if you understood",icon = 'warning')
        if MsgBox == 'no':
            tk.messagebox.showinfo('Your need','Please go through the readme file properly')
        
    elif(is_number(Id) and name.isalpha()):
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                    #incrementing sample number 
                    sampleNum=sampleNum+1
                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage/"+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    #display the frame
                    cv2.imshow('frame',img)
                #wait for 100 miliseconds 
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum>60:
                    break
            cam.release()
            cv2.destroyAllWindows() 
            res = "Images Saved for ID : " + Id +" Name : "+ name
            row = [Id , name]
            with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                if(isheaderrow==0):
                    writer.writerow(['Id', 'Name'])
                    isheaderrow = 1
                writer.writerow(row)
            csvFile.close()
            message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)

def TrainData():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadepath = "haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadepath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/TRAINNER.YML")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    clear1();
    clear2();
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Your model has been trained successfully!!')
    
def getImagesAndLabels(path):
    print(path)
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    
    faces=[]

    Ids=[]

    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids


def TrackData():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel/TRAINNER.YML")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails/StudentDetails.csv", on_bad_lines='skip') 
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 40):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 40):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)
    res = "Attendance Taken"
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Congratulations ! Your attendance has been marked successfully for the day!!')




def Quit_Studywindow():
   MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
   if MsgBox == 'yes':
       tk.messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
       Study.destroy()

takeImg = tk.Button(Study, text="IMAGE CAPTURE BUTTON", command=TakeImages  ,fg="black"  ,bg="blue"  ,width=28  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
takeImg.place(x=200-x_cord, y=425-y_cord)

trainImg = tk.Button(Study, text="MODEL TRAINING BUTTON", command=TrainData ,fg="black"  ,bg="blue"  ,width=28  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
trainImg.place(x=600-x_cord, y=425-y_cord)

trackImg = tk.Button(Study, text="ATTENDANCE MARKING BUTTON", command=TrackData  ,fg="black"  ,bg="blue"  ,width=35  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
trackImg.place(x=1020-x_cord, y=424-y_cord)

StudyquitWindow = tk.Button(Study, text="QUIT", command=Quit_Studywindow  ,fg="black"  ,bg="red"  ,width=10  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
StudyquitWindow.place(x=600, y=725-y_cord)




Study.mainloop()

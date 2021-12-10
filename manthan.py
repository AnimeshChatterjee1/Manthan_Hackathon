import tkinter
from tkinter import *
from tkinter import messagebox
import pickle
from tkinter import filedialog
from PIL import Image,ImageOps
import cv2
import numpy as np
import os
from module1 import detect
import matplotlib.pyplot as plt
import pytesseract
import imutils

def bin_file():
    f = open("username.bin", "ab+")
    try:
        data = pickle.load(f)
        if len(data) == 0:
            name_pass = {}
            pickle.dump(name_pass, f)
            f.close()
        else:
            f.close()
    except:
        name_pass = {}
        pickle.dump(name_pass, f)
        f.close()


bin_file()


def first_page():
    # creating a tkinter window
    tk = tkinter.Tk()
    tk.title("")
    tk.minsize(800, 500)
    tk.maxsize(800, 500)
    # creating a canvas
    canvas = Canvas(tk, width=800, height=500, bg="black")
    # adding text on canvas
    canvas.create_text(380, 50, text="WELCOME", fill="LIGHT GREEN", font=('arial 20 bold'))
    canvas.create_text(55, 200, text="USER ID", fill="RED", font=('Helvetica 15 bold'))
    canvas.create_text(70, 260, text="PASSWORD", fill="BLUE", font=('Helvetica 15 bold'))
    # adding image in application
    img = PhotoImage(file="img.png")
    canvas.create_image(480, 25, anchor=NW, image=img)
    # entry box to collect username and password
    nameentry = Entry(tk)
    passentry = Entry(tk)
    canvas.create_window(200, 200, window=nameentry)
    canvas.create_window(200, 260, window=passentry)
    # adding team logo on the application
    canvas.create_text(380, 490, text="TECH PHANTOMS", fill="ORANGE", font=('arial 10 bold'))

    # creating login button
    def login():
        f = open("username.bin", "rb+")
        data = pickle.load(f)
        f.close()
        if len(data) == 0:
            messagebox.showinfo("ATTENTION", "No User Found\nUse Sign Up")
        else:
            username = data.keys()
            if (nameentry.get()).upper() in username:
                if data[(nameentry.get()).upper()] == passentry.get():
                    # creating the second page now
                    messagebox.showinfo("", f"WELCOME {(nameentry.get()).upper()}")
                    name = (nameentry.get()).upper()
                    tk.destroy()

                    def second_page():
                        f=open("images_path.bin","wb+")
                        f.close()
                        uk = tkinter.Tk()
                        uk.title("")
                        uk.minsize(800, 500)
                        uk.maxsize(800, 500)
                        # creating a canvas
                        canvas2 = Canvas(uk, width=800, height=500, bg="black")
                        canvas2.create_text(350, 50, text=f"HELLO THERE!!\n{name}", fill="YELLOW",
                                            font=('arial 20 bold'))
                        canvas2.create_text(350, 110, text="Please Select Images to be uploaded", fill="cyan",
                                            font=('arial 15 bold'))
                        canvas2.create_text(82, 200, text="UPLOAD IMAGE", fill="lime",
                                            font=('arial 15 bold'))
                        canvas2.create_text(80, 280, text="DELETE IMAGE", fill="red",
                                            font=('arial 15 bold'))
                        canvas2.create_text(380, 490, text="TECH PHANTOMS", fill="orange",
                                            font=('arial 10 bold'))
                        # adding image in application
                        img = PhotoImage(file="img.png")
                        canvas2.create_image(480, 25, anchor=NW, image=img)

                        def upload():
                            f = open("images_path.bin", "ab+")
                            try:
                                data_path = pickle.load(f)
                                if len(data_path) == 0:
                                    l = list()
                                    pickle.dump(l, f)
                                    f.close()
                                else:
                                    f.close()
                            except:
                                l = list()
                                pickle.dump(l, f)
                                f.close()
                            f = open("images_path.bin", "rb+")
                            data_path=pickle.load(f)
                            f.close()
                            if len(data_path) == 1 and len(data_path)<2:
                                messagebox.showinfo("ATTENTION", "Max 1 Image can be selected")
                                data_path = data_path[0]

                            else:
                                name = filedialog.askopenfilenames(filetypes=[("Image Files",
                                                                           ".png .jfif, .jpg, *jpeg"),("Image Files",
                                                                           ".png .jfif, .jpeg, *.jpg")])  # returns a tuple with opened file's complete path
                                f = open("images_path.bin", "rb+")
                                set_list = set()
                                data_path = pickle.load(f)
                                f.close()
                                for i in name:
                                    set_list.add(i)
                                data_path.extend(set_list)
                                set_data_path = set(data_path)
                                data_path = list(set_data_path)
                                f = open("images_path.bin", "wb")
                                pickle.dump(data_path, f)
                                f.close()
                                print(list(data_path))

                        def delete():
                            f=open("images_path.bin","rb+")
                            try:
                                data=pickle.load(f)
                                f.close()
                                messagebox.showinfo("SUCCESS",f"Deleted {data.pop()} Successfully")
                                f=open("images_path.bin","wb")
                                pickle.dump(data,f)
                                f.close()
                            except:
                                messagebox.showinfo("Attention","NO image selected")

                        def show():
                            try:
                                f = open("images_path.bin", "rb")
                                data = pickle.load(f)
                                if len(data)==0:
                                    messagebox.showinfo("ATTENTION","No image selected")
                                else:
                                    image = Image.open(data[0])
                                    image.show()

                            except:
                                messagebox.showinfo("ATTENTION","No image selected yet")
                        def third_page():

                            f=open("images_path.bin","rb")
                            try:
                                data=pickle.load(f)
                                f.close()
                                if len(data)!=0:
                                    uk.destroy()
                                    st = tkinter.Tk()
                                    st.maxsize(800, 500)
                                    st.minsize(800, 500)
                                    canvas3 = Canvas(st, width=800, height=500, bg="black")
                            # adding text on canvas
                                    canvas3.create_text(380, 30, text=f"USER:{name.title()}", fill="LIGHT GREEN",
                                                font=('arial 12 bold'))
                                    count=0

                                    def home():
                                        st.destroy()
                                        second_page()

                                    def detection():
                                        vehicles_folder_count = []
                                        coordinate = []
                                        f=open("images_path.bin","rb")
                                        data=pickle.load(f)

                                        f.close()

                                        class VehicleDetector:

                                            def __init__(self):
                                                # Load Network
                                                net = cv2.dnn.readNet("yolov3.weights", "yo3.cfg")
                                                self.model = cv2.dnn_DetectionModel(net)
                                                self.model.setInputParams(size=(832, 832), scale=1 / 255)

                                                # Allow classes containing Vehicles only
                                                self.classes_allowed = [2, 3, 4, 5, 6, 7]

                                            def detect_vehicles(self, img):
                                                # Detect Objects
                                                vehicles_boxes = []
                                                class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.6)
                                                for class_id, score, box in zip(class_ids, scores, boxes):
                                                    if score < 0.1:
                                                        # Skip detection with low confidence
                                                        continue

                                                    if class_id in self.classes_allowed:
                                                        vehicles_boxes.append(box)

                                                return vehicles_boxes

                                        vd = VehicleDetector()

                                        # enter file address in commas below

                                        img1 = cv2.imread(f"{data[0]}")
                                        img1 = cv2.resize(img1, (600, 600))

                                        vehicle_boxes = vd.detect_vehicles(img1)
                                        vehiclecount = len(vehicle_boxes)

                                        # Update total count
                                        vehicles_folder_count.append(vehiclecount)

                                        for box in vehicle_boxes:
                                            a, b, c, d = box
                                            print(a, b, c, d)
                                            # coordinates of detected object
                                            startpoint = a
                                            startpoint1 = b
                                            endpoint = a + c
                                            endpoint1 = b + d

                                            # appending coordinates of detected object
                                            coordinate.append([startpoint1, endpoint1, startpoint, endpoint])

                                            # drawing rectangle for each car
                                            cv2.rectangle(img1, (a, b), (a + c, b + d), (0, 255, 0), 3)
                                            cv2.putText(img1, "Total: " + str(vehiclecount), (10, 30), 0, 1, (0, 255, 0),
                                                        2)
                                            print("Total: " + str(vehiclecount))
                                        # showing image with rectangles
                                        cv2.imshow("Cars", img1)
                                        cv2.waitKey(0)
                                        print(vehicles_folder_count)
                                        # enter file address in commas below
                                        cv2.destroyAllWindows()
                                    def classification():
                                        f=open("images_path.bin","rb")
                                        data=pickle.load(f)
                                        f.close()
                                        img = cv2.imread(f'{data[0]}')
                                        img=cv2.resize(img,(800,800))
                                        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
                                        frozen_model = 'frozen_inference_graph.pb'
                                        model = cv2.dnn_DetectionModel(frozen_model, config_file)
                                        classLabels = []
                                        file_name = 'Labels.txt'
                                        with open(file_name, 'rt') as fpt:
                                            classLabels = fpt.read().rstrip('\n').split('\n')
                                            # classLabels.append(fpt.read())
                                        # print(classLabels)
                                        print(len(classLabels))
                                        print(classLabels)
                                        model.setInputSize(400, 400)
                                        model.setInputScale(1.0 / 127.5)
                                        model.setInputMean((127.5, 127.5, 127.5))
                                        model.setInputSwapRB(True)

                                        plt.imshow(img)
                                        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                        ClassIndex, confidece, bbox = model.detect(img, confThreshold=0.55)
                                        print(ClassIndex)
                                        font_scale = 1.5
                                        font = cv2.FONT_HERSHEY_PLAIN
                                        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(),
                                                                         bbox):
                                            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                            cv2.rectangle(img, boxes, (255, 0, 0), 2)
                                            # cv2.putText(img, text, (text_offset_x, text_offset y), font, fontScale-font_scale, color-(0, 0, 0), thickness=1)
                                            cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                                        font, fontScale=font_scale, color=(0, 0, 200), thickness=2)
                                        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                        cv2.resize(img,(500,500))
                                        cv2.imshow('abc', img)
                                        cv2.waitKey(0)
                                        

                                    submit6 = Button(st, text="Home Page", command=home)
                                    canvas3.create_window(380,340, window=submit6)
                                    submit7 = Button(st, text="Show Detected Object", command=detection)
                                    canvas3.create_window(380, 260, window=submit7)
                                    submit8 = Button(st, text="Classification", command=classification)
                                    canvas3.create_window(380, 180, window=submit8)
                                    canvas3.pack()
                                    st.mainloop()

                                else:
                                    messagebox.showinfo("Attention", "No image selected")
                            except:
                                messagebox.showinfo("Attention","No image selected")
                        def number_plate():
                            f = open("images_path.bin", "rb")
                            try:
                                data = pickle.load(f)
                                f.close()
                                if len(data) != 0:
                                    img = cv2.imread(f'{data[0]}', cv2.IMREAD_COLOR)
                                    img = cv2.resize(img, (620, 480))
                                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                    gray = cv2.bilateralFilter(gray, 13, 15, 15)
                                    edged = cv2.Canny(gray, 30, 200)
                                    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    contours = imutils.grab_contours(contours)
                                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                                    for c in contours:
                                        peri = cv2.arcLength(c, True)
                                        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                                        if len(approx) == 4:
                                            screenCnt = approx
                                            break
                                        else:
                                            screenCnt = None
                                    if screenCnt is None:
                                        detected = 0
                                        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                                        img = cv2.imread(f'{data[0]}', cv2.IMREAD_COLOR)
                                        img = imutils.resize(img, width=500)
                                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
                                        gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
                                        edged = cv2.Canny(gray, 30, 200)
                                        cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                                                     cv2.CHAIN_APPROX_SIMPLE)
                                        img1 = img.copy()
                                        cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
                                        cv2.imshow("img1", img1)
                                        cv2.waitKey(0)
                                        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
                                        screenCnt = None  # will store the number plate contour
                                        img2 = img.copy()
                                        cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
                                        cv2.imshow("img2", img2)  # top 30 contours
                                        cv2.waitKey(0)

                                    else:
                                        detected = 1
                                    if detected == 1:
                                        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
                                        mask = np.zeros(gray.shape, np.uint8)
                                        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
                                        new_image = cv2.bitwise_and(img, img, mask=mask)
                                        (x, y) = np.where(mask == 255)
                                        (topx, topy) = (np.min(x), np.min(y))
                                        (bottomx, bottomy) = (np.max(x), np.max(y))
                                        cv2.imshow("", cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3))
                                        cv2.waitKey(0)
                                        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
                                        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                                        cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
                                        read = pytesseract.image_to_string(Cropped)
                                        read = ''.join(e for e in read if e.isalnum())
                                        print(read)
                                        cv2.imshow("hello", Cropped)
                                        cv2.waitKey(0)
                                        cv2.destroyAllWindows()


                                else:
                                    messagebox.showinfo("Attention", "no image selected")
                            except:
                                messagebox.showinfo("Attention","No image selected")






                        submit2 = Button(uk, text="click here", command=upload)
                        canvas2.create_window(200, 200, window=submit2)
                        submit3 = Button(uk, text="click here", command=delete)
                        canvas2.create_window(200, 280, window=submit3)
                        submit4 = Button(uk, text="View selected IMAGE", command=show)
                        canvas2.create_window(200, 340, window=submit4)
                        submit5 = Button(uk, text="Process", command=third_page)
                        canvas2.create_window(200, 400, window=submit5)
                        submit6=Button(uk,text="Number plate detection",command=number_plate)
                        canvas2.create_window(400,400, window=submit6)

                        canvas2.pack()
                        uk.mainloop()

                    second_page()
                    pass
                else:
                    messagebox.showinfo("ACCESS DENIED", "ENTER PASSWORD PROPERLY")
            else:
                messagebox.showinfo("NOT FOUND", "user id not found")

    submit1 = Button(tk, text="      Login      ", command=login)
    canvas.create_window(200, 320, window=submit1)
    # new user id password
    canvas.create_text(50, 400, text="New User?", fill="magenta", font=('arial 12 bold'))

    # creating signup button
    def signup():
        tk.destroy()
        fk = tkinter.Tk()
        fk.title("SIGN UP")
        fk.minsize(300, 200)
        fk.maxsize(300, 200)
        # creating canvas for sing up window
        canvas1 = Canvas(fk, width=300, height=200, bg="silver")
        canvas1.create_text(60, 70, text="USER ID", fill="RED", font=('Helvetica 15 bold'))
        canvas1.create_text(70, 140, text="PASSWORD", fill="BLUE", font=('Helvetica 15 bold'))
        nameentry1 = Entry(fk)
        passentry1 = Entry(fk)
        canvas1.create_window(200, 70, window=nameentry1)
        canvas1.create_window(200, 140, window=passentry1)
        f = open("username.bin", "rb+")
        data = pickle.load(f)
        f.close()

        def confirm():
            if len(nameentry1.get()) > 0:
                if (nameentry1.get()).upper() not in data.keys():
                    if len(passentry1.get()) > 0:
                        d = {(nameentry1.get()).upper(): passentry1.get()}
                        data.update(d)
                        f = open("username.bin", "wb+")
                        pickle.dump(data, f)
                        f.close()
                        messagebox.showinfo("success", "user successfully created")
                        fk.destroy()
                        first_page()
                    else:
                        messagebox.showinfo('error', "please enter password")
                else:
                    messagebox.showinfo("ops!", "username already exits")
            else:
                messagebox.showinfo("error", "please enter username")

        submit1 = Button(fk, text="CONFIRM", command=confirm)
        canvas1.create_window(260, 180, window=submit1)
        canvas1.pack()
        fk.mainloop()

    submit2 = Button(tk, text="      SIGN UP      ", command=signup)
    canvas.create_window(150, 400, window=submit2)

    canvas.pack()
    tk.mainloop()


first_page()

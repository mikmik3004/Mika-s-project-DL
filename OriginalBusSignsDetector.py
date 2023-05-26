#!/usr/bin/env python
# coding: utf-8

import sys
import subprocess
#subprocess.run(["pip", "install", "pytesseract"])
#subprocess.run(["pip", "install", "deskew"])
import numpy as np
import math
from typing import Tuple, Union
import cv2
from deskew import determine_skew
import glob
import pytesseract
import shutil
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QRect, Qt


def model(image_file_name, MainWindow):

    loc = image_file_name.rfind("/")
    extracted_name = image_file_name[loc+1:len(image_file_name)]
    cropped_image_name = f'./runs/detect/predict/crops/bus-signs/{extracted_name}'
  
    def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    def deskew(_img):
        image = cv2.imread(_img)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        rotated = rotate(image, angle, (0, 0, 0))
        return rotated.astype(np.uint8)

    def detect_bus_signs_numbers(new_width, rotate=False):
        successes = 0
        failed = 0
        for image_path_in_colab in glob.glob(f'{cropped_image_name}')[:1]:
            # load the input image, convert it from BGR to RGB channel ordering,
            # and initialize our Tesseract OCR options as an empty string
            image = cv2.imread(image_path_in_colab)

            # Rotate / Deskew the image
            if rotate:
                image = deskew(image_path_in_colab)

            image_proportsion =  image.shape[0] / image.shape[1]

            #Rescaling
            dim = (new_width, int(new_width*image_proportsion))
            #dim = (new_width, new_width)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

            height = image.shape[0]
            width = image.shape[1]

            #Grey Scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #dilasion and erosion
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.dilate(image, kernel, iterations=0)
            image = cv2.erode(image, kernel, iterations=1)

            #filters
            image = cv2.adaptiveThreshold(cv2.GaussianBlur(image, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            #image = cv2.adaptiveThreshold(cv2.bilateralFilter(image, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            image = cv2.adaptiveThreshold(cv2.medianBlur(image, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

            #Rectangle
            image = abs(255-image)
            rectangle = np.zeros((height, width), dtype = image.dtype)
            rectangle = cv2.rectangle(rectangle, (width//3, height//6), ((width-(2*width)//7), 2*height//7), 255, -1)
            image = cv2.bitwise_and(image, rectangle)
            image = abs(255-image)

            options = "--psm 11 -c tessedit_char_whitelist='1234567890'"
            detected = pytesseract.image_to_string(image, config=options)
            numeric_filter = filter(str.isdigit, detected)
            digits = "".join(numeric_filter)
            
        return digits, cropped_image_name

    pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'

    shutil.rmtree(f'./runs/detect/predict', ignore_errors=True)

    model = YOLO("BusSignesModel.pt")
    results = model.predict(image_file_name, task="detect",conf=0.6, save=True, save_crop=True)

    detect_digits, cropped_image =  detect_bus_signs_numbers(440, rotate=True)
  
    return detect_digits, cropped_image


#The main class of the GUI application
class Ui_MainWindow(QDialog):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("Mika Bus Signs Detection")
        MainWindow.resize(1137, 925)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.boutton1 = QPushButton(self.centralwidget)
        self.boutton1.setObjectName(u"boutton1")
        self.boutton1.setGeometry(QRect(460, 800, 211, 71))
        self.lineEdit1 = QLineEdit(self.centralwidget)
        self.lineEdit1.setObjectName(u"lineEdit1")
        self.lineEdit1.setGeometry(QRect(20, 750, 1091, 31))
        self.left_picture = QLabel(self.centralwidget)
        self.left_picture.setObjectName(u"left_picture")
        self.left_picture.setGeometry(QRect(110, 160, 341, 431))
        self.left_picture.setFrameShape(QFrame.Box)
        self.left_picture.setMidLineWidth(0)
        self.left_picture.setMargin(8)
        self.right_picture = QLabel(self.centralwidget)
        self.right_picture.setObjectName(u"right_picture")
        self.right_picture.setGeometry(QRect(600, 50, 491, 631))
        self.right_picture.setFrameShape(QFrame.Box)
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(560, 20, 20, 721))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(220, 130, 150, 20))
        self.label.setIndent(-1)
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(790, 20, 150, 20))
        self.label_2.setIndent(-1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 1137, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.boutton1.clicked.connect(self.browsefiles)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Buys Station Detector - By Mika Yeshaayaou"))
        self.boutton1.setText(_translate("MainWindow", "Upload Image"))
        self.left_picture.setText("")
        self.right_picture.setText("")
        self.label.setText(_translate("MainWindow", u"The Corped Image", None))
        self.label_2.setText(_translate("MainWindow", u"The Selected Image", None))

        
    # openning a file browser that returns a selected image (.jpg) file
    # and then call the "model" function to detect the bus sign using 
    # my YOLO model, and then detect the digits using OCR
    # then prints the cropped file and the detected digits on the UI
    def browsefiles(self):
        fname=QFileDialog.getOpenFileName(self, 'Open file', '', 'Images (*.png, *.xmp *.jpg)')
        self.lineEdit1.setText("Detecting...")
        self.lineEdit1.repaint()

        image_file_name = fname[0]

        detect_digits,cropped_image = model(image_file_name, MainWindow)

        pixmap = QPixmap(cropped_image)
        self.left_picture.setPixmap(pixmap)
        self.left_picture.updateGeometry()
        full_pixmap = QPixmap(image_file_name)
        full_pixmap = full_pixmap.scaled(self.right_picture.height(), self.right_picture.width(),  Qt.KeepAspectRatio)
        self.right_picture.setPixmap(full_pixmap)
        self.right_picture.updateGeometry()

        self.lineEdit1.setText("Detected Bus Station: " + detect_digits)
        self.lineEdit1.repaint()
    # retranslateUi

#the starting commands for starting the GUI application
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
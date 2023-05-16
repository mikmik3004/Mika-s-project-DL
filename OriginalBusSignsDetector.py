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

def model(image_file_name, MainWindow):

    DEBUG=False
    loc = image_file_name.rfind("/")
    extracted_name = image_file_name[loc+1:len(image_file_name)]
    cropped_image_name = f'./runs/detect/predict/crops/bus-signs/{extracted_name}'
  
    if DEBUG:
        print("Loading Image=" + image_file_name)
        print("extracted name:", extracted_name)
        print("Expected Croped Image:" + cropped_image_name)

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
            if DEBUG:
                print("Processing Image:", image_path_in_colab)

            # load the input image, convert it from BGR to RGB channel ordering,
            # and initialize our Tesseract OCR options as an empty string
            image = cv2.imread(image_path_in_colab)

            # Rotate / Deskew the image
            if rotate:
                if DEBUG:
                    print("image before deskew:")
                cv2.imshow("before deskew", image)

                image = deskew(image_path_in_colab)
                if DEBUG:
                    print("image after deskew:")
                cv2.imshow("after deskew", image)

            image_proportsion =  image.shape[0] / image.shape[1]

            if DEBUG:
                print("Original Size: Width:",image.shape[1], "Height:", image.shape[0], "Proportion:", image_proportsion, "New Width:", new_width)

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
            
            if DEBUG:
                print("detected digits:", digits)

        return digits, cropped_image_name

    #Need to install Tesseract-OCR on the specified folder ("C:/Program Files")
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    shutil.rmtree(f'./runs/detect/predict', ignore_errors=True)
    subprocess.run(["yolo", "task=detect", "mode=predict", f"model=./BusSignesModel.pt", "conf=0.6", f"source={image_file_name}", "save=True", "save_crop=True"])

    detect_digits, cropped_image =  detect_bus_signs_numbers(440, rotate=True)
  
    if DEBUG:  
        print("Done. detect=", detect_digits)

    return detect_digits, cropped_image


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap


class Ui_MainWindow(QDialog):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Mika Bus Signs Detection")
        MainWindow.resize(611, 505)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.boutton1 = QtWidgets.QPushButton(self.centralwidget)
        self.boutton1.setGeometry(QtCore.QRect(230, 380, 151, 71))
        self.boutton1.setObjectName("boutton1")

        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(250, 40, 171, 41))
        self.label1.setText("")
        self.label1.setObjectName("label1")
        self.lineEdit1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit1.setGeometry(QtCore.QRect(30, 320, 541, 31))
        self.lineEdit1.setObjectName("lineEdit1")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(130, 50, 341, 201))
        self.label.setText("")
        self.label.setObjectName("label")
        label = QLabel(self)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 611, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.boutton1.clicked.connect(self.browsefiles)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.boutton1.setText(_translate("MainWindow", "Click to uplode a picture"))

    def browsefiles(self):
        fname=QFileDialog.getOpenFileName(self, 'Open file', '', 'Images (*.png, *.xmp *.jpg)')
        self.lineEdit1.setText("Detecting...")
        self.lineEdit1.repaint()

        image_file_name = fname[0]

        detect_digits,cropped_image = model(image_file_name, MainWindow)

        self.lineEdit1.setText(detect_digits)
        pixmap = QPixmap(cropped_image)
        self.label.setPixmap(pixmap)
        self.label.updateGeometry()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

from PyQt5.QtWidgets import *
import sys,pickle
import os

from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QDir, Qt, QSortFilterProxyModel
from PyQt5.QtWidgets import QDialog ,QApplication, QFileDialog, QWidget, QTextEdit, QLabel
from PyQt5.uic import loadUi
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtGui import QImage
import cv2, imutils
from einops import rearrange, reduce, repeat
from lenet5 import *
import numpy as np


main_path = os.path.dirname(os.path.abspath(__file__)) #file path main.py
work_path = os.path.split(main_path) #path working folder (whole file project)
ui_folder = os.path.join(main_path,"ui/") #ui_folder path


class error_window(QMainWindow):
    def __init__(self):
        super(error_window, self).__init__()
        

class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        uic.loadUi(os.path.join(ui_folder,"main2.ui"), self)
        
        self.filePath = None
        self.methods = ["adam", "rmsprop"]
        self.learningRate = ["0.001", "0.0001"]
        self.batch = ["32"]
        self.epochs = ["101", "151", "201"]
        
        self.output = None
        
        self.optimizerCombo.addItems(self.methods)
        self.learningRateCombo.addItems(self.learningRate)
        self.epochsCombo.addItems(self.epochs)
        self.batchCombo.addItems(self.batch)
        
        self.lenet = None
        if self.lenet == None:
            self.modelLabel.setText("No Model")
            
        self.openImageBtn.clicked.connect(self.browseImage)
        self.loadModelBtn.clicked.connect(self.browseModel)
        self.recogImageBtn.clicked.connect(self.predictImage)
        imagePath = "data_jepun"
        self.data = Data(main_path, imagePath)
        self.label = self.data.loadLabel()


    def browseImage(self):
        self.filePath = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        _, self.fname = os.path.split(self.filePath)
        self.textFname.setText(self.fname)
        print(self.filePath) 
        self.image = cv2.imread(self.filePath)
        self.setPhoto(self.image)

        
    def setPhoto(self,image):
        """ This function will take image input and resize it 
			only for display purpose and convert it to QImage
			to set at the label.
		"""
        self.tmp = image
        image = imutils.resize(image,width=300)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.imageSpace.setPixmap(QtGui.QPixmap.fromImage(image))
        
    
    def browseModel(self):
        
        method = self.optimizerCombo.currentText()
        learningRate = self.learningRateCombo.currentText()
        epochs = self.epochsCombo.currentText()
        batch = self.batchCombo.currentText()
        print(method, learningRate, epochs, batch)
        self.lenet = LENET5([], [], [], [], method=method,epochs=epochs, batch=batch, learningRate=learningRate )
        self.lenet.load_parameters(mainPath=main_path,epochs=epochs,method=method, batch=batch, learningRate=learningRate)
        if self.lenet != None:
            self.output = self.modelLabel.setText("Model Loaded")
    
    def predictImage(self):
        self.output = self.lenet.one_image(self.lenet.layers, self.filePath)

        indeks = np.argmax(self.output)

        self.predLabel.setText(self.label[indeks])
        self.probLabel.setText(str(self.output[0,indeks]))



app = QtWidgets.QApplication(sys.argv)
window = App()
widget = QtWidgets.QStackedWidget()
widget.addWidget(window)
#widget.setFixedWidth(500)
#widget.setFixedHeight(500)
widget.show()
app.exec_()
#sys.exit( app.exec_() )



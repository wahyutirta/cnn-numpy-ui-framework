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

import matplotlib as plt
plt.use('Qt5Agg')
#matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker


main_path = os.path.dirname(os.path.abspath(__file__)) #file path main.py
work_path = os.path.split(main_path) #path working folder (whole file project)
ui_folder = os.path.join(main_path,"ui/") #ui_folder path


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        
        super(MplCanvas, self).__init__(self.fig)
        #self.fig.tight_layout()
        
		


class error_window(QMainWindow):
    def __init__(self):
        super(error_window, self).__init__()


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        self.ui = uic.loadUi(os.path.join(ui_folder,"main2.ui"), self)
        
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
        
        self.optimizerCombo.currentIndexChanged.connect(self.resetModel)
        self.learningRateCombo.currentIndexChanged.connect(self.resetModel)
        self.epochsCombo.currentIndexChanged.connect(self.resetModel)
        self.batchCombo.currentIndexChanged.connect(self.resetModel)
        
        
    def resetModel(self):
        self.lenet = None
        
        if self.lenet == None:
            self.output = self.modelLabel.setText("No Model")
            print("model null")
        

    def browseImage(self):
        self.filePath = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        _, self.fname = os.path.split(self.filePath)
        self.textFname.setText(self.fname)
        print(self.filePath) 
        self.image = cv2.imread(self.filePath)
        self.setPhoto(self.image)
        
        #clear canvas
        self.canvas1 = MplCanvas(self, width=4, height=6, dpi=100)
        self.ui.gridLayout_4.addWidget(self.canvas1, 1, 6, 1, 1)
        self.canvas1.fig.clf()
        
        self.canvas2 = MplCanvas(self, width=4, height=6, dpi=100)
        self.ui.gridLayout_4.addWidget(self.canvas2, 1, 7, 1, 1)
        self.canvas2.fig.clf()


        
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
        self.lenet = LENET5( method = method, epochs = epochs, batch = batch, learningRate = learningRate) 
        
        self.lenet.load_parameters(mainPath=main_path,epochs=epochs,method=method, batch=batch, learningRate=learningRate)
        if self.lenet != None:
            self.output = self.modelLabel.setText("Model Loaded")
    
    def predictImage(self):
        self.output = self.lenet.one_image(self.lenet.layers, self.filePath)

        indeks = np.argmax(self.output)

        self.predLabel.setText(self.label[indeks])
        pribability = str(self.output[0,indeks] *  100)
        self.probLabel.setText(str(pribability + "%"))
        
        features1 = self.lenet.displayFeature(self.lenet.layers, self.filePath, 1)
        features1 = features1.astype(np.uint8)
        self.features1 = features1
        
        features2 = self.lenet.displayFeature(self.lenet.layers, self.filePath, 2)
        features2 = features2.astype(np.uint8)
        self.canvasManager(features1,features2)
    
    def canvasManager(self,features1, features2):
        
        self.canvas1 = MplCanvas(self, width=4, height=6, dpi=100)
        self.ui.gridLayout_4.addWidget(self.canvas1, 1, 6, 1, 1)
        App.plot(self.canvas1,features1)
        
        self.canvas2 = MplCanvas(self, width=4, height=6, dpi=100)
        self.ui.gridLayout_4.addWidget(self.canvas2, 1, 7, 1, 1)
        App.plot(self.canvas2,features2)

        """
        rows = 3
        columns = 2
        counter = 1
        print(features.shape)
        for feature in features:
            
            print(feature)
            title = str("feature " + str(counter))
            self.canvas.axes = self.canvas.fig.add_subplot(rows, columns, counter)
            
            
            self.canvas.axes.imshow(feature)
            self.canvas.axes.axis("off")
            self.canvas.axes.set_title(title)
            counter += 1
            
        self.canvas.draw()
        """
    @staticmethod
    def plot(canvas,features):

        rows = 3
        columns = 2
        counter = 1
        print(features.shape)
        for feature in features:
            
            print(feature)
            title = str("feature " + str(counter))
            canvas.axes = canvas.fig.add_subplot(rows, columns, counter)
            
            
            canvas.axes.imshow(feature)
            canvas.axes.axis("off")
            canvas.axes.set_title(title)
            counter += 1
            
        canvas.draw()


app = QtWidgets.QApplication(sys.argv)
window = App()
widget = QtWidgets.QStackedWidget()
widget.addWidget(window)
widget.setFixedWidth(1070)
widget.setFixedHeight(660)
widget.show()
app.exec_()
#sys.exit( app.exec_() )



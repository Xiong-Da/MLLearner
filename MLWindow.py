import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtWidgets import QPushButton,QWidget,QLabel,QSlider
from PyQt5.QtWidgets import QHBoxLayout,QVBoxLayout,QFileDialog,QDialog\
    ,QListWidget,QListWidgetItem,QInputDialog
from PyQt5.QtGui import QImage,QPixmap,QFont
from PyQt5.QtCore import Qt,pyqtSignal,QTimer

import ImageProcess
import Classfier

class ConfigDialog(QDialog):

    starting=pyqtSignal()

    def __init__(self,parent):
        super().__init__(parent)
        self.listWidget=QListWidget()
        self.listWidget.setSelectionMode(2)

        self.add=QPushButton("add")
        self.insert=QPushButton("insert")
        self.delete=QPushButton("delete")
        self.start=QPushButton("start")

        self.initUI()
        self.initAction()

    def initUI(self):
        bottom=QHBoxLayout()
        bottom.addWidget(self.add)
        bottom.addWidget(self.insert)
        bottom.addWidget(self.delete)
        bottom.addWidget(self.start)

        layout=QVBoxLayout()
        layout.addWidget(self.listWidget)
        layout.addLayout(bottom)

        self.setLayout(layout)
        self.resize(300,300)

    def getItem(self):
        num, ok = QInputDialog.getInt(self, "标题", "计数:", 2, -1000, 1000, 1)
        if ok!=True:return None

        item=QListWidgetItem(str(num))
        item.setFont(QFont("",15))

        return item

    def onDelete(self):
        items=self.listWidget.selectedItems()
        for item in items:
            self.listWidget.takeItem(self.listWidget.row(item))

    def onAdd(self):
        item=self.getItem()
        if item==None:return

        self.listWidget.addItem(item)

    def onInsert(self):
        item = self.getItem()
        if item == None: return

        row = self.listWidget.currentRow()
        self.listWidget.insertItem(row, item)

    def onStart(self):
        self.close()
        self.starting.emit()

    def initAction(self):
        self.delete.clicked.connect(self.onDelete)
        self.add.clicked.connect(self.onAdd)
        self.insert.clicked.connect(self.onInsert)
        self.start.clicked.connect(self.onStart)

    def getNNParam(self):
        param=[]
        for i in range(self.listWidget.count()):
            item=self.listWidget.item(i)
            data=int(item.text())
            param.append(data)
        return param

class MLWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.timer=QTimer(self)

        self.configDialog=ConfigDialog(self)

        self.threshhold=200
        self.dataImgae=None
        self.rawData=None
        self.outputPatten=None
        self.message= "None"

        self.threshholdSlider=None
        self.fileButton=None
        self.messageLabel=None
        self.nnConfigButton=None

        self.pattenImage=None
        self.nnImage=None

        self.initUI()
        self.initAction()

    def initUI(self):
        self.threshholdSlider=QSlider(Qt.Horizontal)
        self.threshholdSlider.setMinimum(0)
        self.threshholdSlider.setMaximum(255)
        self.threshholdSlider.setValue(self.threshhold)

        self.fileButton=QPushButton("file")
        self.nnConfigButton=QPushButton("config")

        self.messageLabel=QLabel("Message:None")
        self.pattenImage=QLabel("None")
        self.nnImage=QLabel("None")

        leftBottom=QHBoxLayout()
        rightBottom=QHBoxLayout()

        left=QVBoxLayout()
        right=QVBoxLayout()

        main=QHBoxLayout()

        leftBottom.addWidget(self.threshholdSlider)
        leftBottom.addWidget(self.fileButton)

        left.addStretch()
        left.addWidget(self.pattenImage,1,Qt.AlignCenter)
        left.addStretch()
        left.addLayout(leftBottom)

        rightBottom.addWidget(self.messageLabel,1,Qt.AlignCenter)
        rightBottom.addWidget(self.nnConfigButton)

        right.addStretch()
        right.addWidget(self.nnImage,1,Qt.AlignCenter)
        right.addStretch()
        right.addLayout(rightBottom)

        main.addLayout(left,1)
        main.addLayout(right,1)

        self.setLayout(main)

        self.resize(800,600)

    def getPixmapFrom(self,image):
        shape=image.shape
        qImage=QImage(image.data,shape[0],shape[1],QImage.Format_Grayscale8)
        return QPixmap.fromImage(qImage)

    def updateDataImage(self):
        if self.rawData is None:return
        self.dataImgae=ImageProcess.binaryImage(self.rawData,self.threshhold)
        self.pattenImage.setPixmap(self.getPixmapFrom(self.dataImgae))

    def onSliderChange(self,value):
        self.threshhold=value
        self.updateDataImage()

    def onFile(self):
        path=QFileDialog.getOpenFileName(self,"select patten")
        if len(path[0])<2:return   #stupid API

        self.rawData=ImageProcess.readImage(path[0])
        self.updateDataImage()

    def onConfig(self):
        self.configDialog.show()

    def startNN(self):
        param=self.configDialog.getNNParam()

        if len(param)==0 or self.dataImgae==None:return

        Classfier.startNN(self.dataImgae,param,self)

    def setOutputPatten(self,patten,message="None"):
        self.outputPatten=patten #Atom operation
        self.message=message

    def showOutputPatten(self):
        patten=self.outputPatten  #Atom operation
        if patten is None:return

        self.nnImage.setPixmap(self.getPixmapFrom(patten))

        message=self.message
        if message==None:return
        self.messageLabel.setText(message)

    def initAction(self):
        self.threshholdSlider.valueChanged.connect(self.onSliderChange)
        self.fileButton.clicked.connect(self.onFile)
        self.nnConfigButton.clicked.connect(self.onConfig)
        self.configDialog.starting.connect(self.startNN)

        self.timer.timeout.connect(self.showOutputPatten)
        self.timer.start(100)


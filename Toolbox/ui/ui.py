from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsOpacityEffect


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1500, 800)

        self.slidern_1 = QtWidgets.QSlider(Qt.Horizontal, Form)
        self.slidern_1.setGeometry(QtCore.QRect(200, 20, 120, 40))
        self.slidern_1.setObjectName("slider_1")
        self.slidern_1.setValue(49)

        self.graphicsView_1 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_1.setGeometry(QtCore.QRect(50, 360, 384, 384))
        self.graphicsView_1.setObjectName("graphicsView_1")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(467, 260,  256, 256))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_3 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_3.setGeometry(QtCore.QRect(467, 600,  256, 256))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_4 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_4.setGeometry(QtCore.QRect(750, 360,  384, 384))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.graphicsView_5 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_5.setGeometry(QtCore.QRect(1167, 260,  256, 256))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.graphicsView_6 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_6.setGeometry(QtCore.QRect(1167, 600,  256, 256))
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.graphicsView_7 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_7.setGeometry(QtCore.QRect(1450, 360,  384, 384))
        self.graphicsView_7.setObjectName("graphicsView_7")

        self.pushButton_1 = QtWidgets.QPushButton(Form)
        self.pushButton_1.setGeometry(QtCore.QRect(50, 20, 120, 40))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 75, 120, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 130, 120, 40))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(200, 130, 120, 40))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(200, 75, 120, 40))
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(360, 20, 120, 120))
        self.pushButton_6.setObjectName("pushButton_6")

        self.pushButton_7 = QtWidgets.QPushButton(Form)
        self.pushButton_7.setGeometry(QtCore.QRect(500, 20, 120, 120))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(Form)
        self.pushButton_8.setGeometry(QtCore.QRect(640, 20, 120, 120))
        self.pushButton_8.setObjectName("pushButton_8")

        self.retranslateUi(Form)
        self.pushButton_1.clicked.connect(Form.open)
        self.pushButton_2.clicked.connect(Form.open_ref)
        self.pushButton_3.clicked.connect(Form.save_img)
        self.pushButton_4.clicked.connect(Form.clear)
        self.pushButton_5.clicked.connect(Form.apply)
        self.pushButton_6.clicked.connect(Form.celeb1)
        self.pushButton_7.clicked.connect(Form.celeb2)
        self.pushButton_8.clicked.connect(Form.celeb3)

        self.slidern_1.valueChanged.connect(Form.slider)

        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "FacialGAN"))
        self.pushButton_1.setText(_translate("Form", "Open Image"))
        self.pushButton_2.setText(_translate("Form", "Open Ref"))
        self.pushButton_3.setText(_translate("Form", "Save Image"))
        self.pushButton_4.setText(_translate("Form", "Clear"))
        self.pushButton_5.setText(_translate("Form", "Apply"))
        self.pushButton_6.setText(_translate("Form", ""))
        self.pushButton_7.setText(_translate("Form", ""))
        self.pushButton_8.setText(_translate("Form", ""))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

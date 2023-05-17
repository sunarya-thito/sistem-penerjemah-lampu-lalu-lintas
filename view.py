# View dari Pendeteksi Lampu Lalu Lintas
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QDialog, QWidget, QLabel
from PyQt5.uic import loadUi


def wrap_label(label: QLabel):
    view = PreviewView(label.window())
    view.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
    view.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, True)
    label.view = view

    def event(e):
        pixmap: QPixmap = label.pixmap()
        if pixmap is None:
            return
        size = pixmap.size()
        ratio = size.width() / size.height()
        window_size = view.size()
        view.resize(QSize(window_size.height() * ratio, window_size.height()))
        view.label.setPixmap(pixmap)
        view.show()

    label.mouseDoubleClickEvent = event


class PreviewView(QDialog):
    def __init__(self, parent):
        super(PreviewView, self).__init__(parent=parent)
        loadUi('preview.ui', self)


class TrafficLightDetectorView(QMainWindow):
    def __init__(self, controller):
        super(TrafficLightDetectorView, self).__init__()
        self.controller = controller
        loadUi('main_ui.ui', self)

        self.show_res.stateChanged.connect(self.controller.handle_show_result)
        self.show_proc.stateChanged.connect(self.controller.handle_show_console)
        self.load_button.clicked.connect(self.controller.load_image)
        self.exec_button.clicked.connect(self.controller.process_image)


class ConsoleView(QDialog):
    def __init__(self, controller, parent):
        super(ConsoleView, self).__init__(parent=parent)
        self.controller = controller
        loadUi('proc_ui.ui', self)

        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.listViewModel = QtGui.QStandardItemModel()
        self.listView.setModel(self.listViewModel)
        self.listView.clicked.connect(parent.controller.handle_console_click)

        wrap_label(self.preview)


class DetectView(QWidget):
    def __init__(self, controller):
        super(DetectView, self).__init__()
        self.controller = controller
        loadUi('detect_ui.ui', self)

        wrap_label(self.label)


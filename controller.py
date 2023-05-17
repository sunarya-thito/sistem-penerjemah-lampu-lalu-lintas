import os

import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel

import detector
import modul
from view import TrafficLightDetectorView, ConsoleView, DetectView


class ProcessBatch:
    def __init__(self, processes, tampilkan_gambar):
        self.processes = processes
        self.tampilkan_gambar = tampilkan_gambar
        self.index_process = -1


def label_set_image(label, cv2_image):
    if cv2_image is None:
        return

    qformat = QtGui.QImage.Format_Indexed8
    if len(cv2_image.shape) == 3:
        if cv2_image.shape[2] == 4:
            qformat = QtGui.QImage.Format_RGBA8888
        else:
            qformat = QtGui.QImage.Format_RGB888

    image = QtGui.QImage(cv2_image, cv2_image.shape[1], cv2_image.shape[0], cv2_image.strides[0], qformat)
    image = image.rgbSwapped()
    label.setPixmap(QtGui.QPixmap.fromImage(image))


class TrafficLightDetectorController:
    def __init__(self):
        self.view = TrafficLightDetectorView(self)
        self.consoleController = ConsoleController(self.view)
        self.process = None
        self.image = None
        self.proc_img = None
        self.processes = []
        self.handler_gambar = []
        modul.muat_cascade()

    def print(self, text, image):
        print(text)
        item = QtGui.QStandardItem(text)
        # item: add on click listener
        item.image = image

        item.setEditable(False)
        item.setSelectable(True)
        self.consoleController.view.listViewModel.appendRow(item)

    def push_process(self, processes, tampilkan_gambar):
        self.processes.append(ProcessBatch(processes, tampilkan_gambar))
        self.process_image()

    def remove_tail_process(self):
        self.processes.pop()

    def load_image(self):
        home_dir = os.path.dirname(os.path.realpath(__file__))
        path = QFileDialog.getOpenFileName(self.view, 'Muat Gambar', home_dir, 'Image files (*.jpg *.png)')[0]
        if path == '':
            return
        img = cv2.imread(path)
        self.image = img
        self.processes.clear()
        self.consoleController.clear_detection()
        self.set_image(img)
        self.push_process(detector.proses_cari_lokasi(self, img), self.tampilkan_gambar)

    def tampilkan_gambar(self, image):
        self.proc_img = image
        if self.is_show_result():
            self.set_image(image)

    def set_image(self, image):
        label_set_image(self.view.label, image)

    def handle_console_click(self, index):
        item = self.consoleController.view.listViewModel.itemFromIndex(index)
        if item.image is None:
            return
        ratio = item.image.shape[1] / item.image.shape[0]
        preview: QLabel = self.consoleController.view.preview
        height = 200
        width = height * ratio
        preview.setFixedWidth(width)
        preview.setFixedHeight(height)
        preview.setMinimumHeight(height)
        preview.setMaximumHeight(height)
        label_set_image(self.consoleController.view.preview, item.image)

    def process_image(self):
        if len(self.processes) == 0:
            # show error
            QtWidgets.QMessageBox.critical(self.view, 'Error', 'Tidak ada gambar yang dimuat')
            return

        previous_process = self.get_current_process()
        self.processes[-1].index_process += 1
        current_process = self.get_current_process()

        current_process.execute_process(previous_process, self.processes[-1].tampilkan_gambar)
        self.print('Proses: ' + current_process.to_string(), current_process.image)
        ratio = current_process.image.shape[1] / current_process.image.shape[0]
        preview = self.consoleController.view.preview
        height = 200
        width = height * ratio
        preview.setFixedWidth(width)
        preview.setFixedHeight(height)
        preview.setMinimumHeight(height)
        preview.setMaximumHeight(height)
        label_set_image(preview, current_process.image)
        if self.processes[-1].index_process + 1 >= len(self.processes[-1].processes):
            self.remove_tail_process()
            if len(self.processes) == 0:
                QtWidgets.QMessageBox.information(self.view, 'Info', 'Proses selesai')
                self.print('Proses selesai', None)
                return
            return
        if not self.is_show_step():
            self.process_image()


    def get_current_process(self):
        if len(self.processes) == 0:
            return None
        index = self.processes[-1].index_process
        if index >= len(self.processes[-1].processes):
            return None
        if index < 0:
            return None
        return self.processes[-1].processes[index]

    def is_show_step(self):
        return self.view.show_step.isChecked()

    def is_show_result(self):
        return self.view.show_res.isChecked()

    def handle_show_result(self, state):
        current_process_image = self.proc_img
        if state == 0 or current_process_image is None:
            self.set_image(self.image)
        else:
            self.set_image(current_process_image)


    def handle_show_console(self, state):
        if state == 0:
            self.consoleController.hide()
        else:
            self.consoleController.show()

    def show(self):
        self.view.show()


class ConsoleController:
    def __init__(self, parent):
        self.view = ConsoleView(self, parent)
        self.detections = []

    def show(self):
        self.view.show()

    def hide(self):
        self.view.hide()

    def add_detection(self, detection):
        self.detections.append(detection)
        self.add_widget(detection.view)

    def clear_detection(self):
        self.detections = []
        self.view.listWidget.clear()
        self.view.listViewModel.clear()

    def add_widget(self, widget):
        list = self.view.listWidget
        item = QtWidgets.QListWidgetItem(parent=list)
        item.setSizeHint(widget.size())
        list.addItem(item)
        list.setItemWidget(item, widget)

class DetectController:
    def __init__(self, parent, initialImage, initialText):
        self.parent = parent
        self.view = DetectView(self)
        self.set_image(initialImage)
        self.view.label_2.setText(initialText)

    def set_image(self, cv2_image):
        if cv2_image is None:
            return
        # self.parent.tampilkan_gambar(cv2_image)
        label = self.view.label

        qformat = QtGui.QImage.Format_Indexed8
        if len(cv2_image.shape) == 3:
            if cv2_image.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        # cv2_image = np.transpose(cv2_image, (1, 0, 2)).copy()
        image = QtGui.QImage(cv2_image, cv2_image.shape[1], cv2_image.shape[0], cv2_image.strides[0], qformat)
        image = image.rgbSwapped()
        label.setPixmap(QtGui.QPixmap.fromImage(image))
import sys

from PyQt5 import QtWidgets

from controller import TrafficLightDetectorController

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainController = TrafficLightDetectorController()
    mainController.show()
    sys.exit(app.exec_())
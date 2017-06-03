import sys
from PyQt5.QtWidgets import  QApplication

import MLWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w=MLWindow.MLWindow()
    w.show()

    sys.exit(app.exec_())
import sys

from PyQt5.QtWidgets import QApplication

from ui import DocumentUploader



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DocumentUploader()
    window.show()
    app.exec_()
    #sys.exit(app.exec_())

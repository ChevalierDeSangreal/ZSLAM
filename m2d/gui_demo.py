from gui.mapcreator import draw_grid, draw_new_circle
from gui import mainwindow, drawwindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import sys
from map import Map
import torch


if __name__ == "__main__":
    map = Map(10., 10., ratio=0.01)
    map.add_circle(1., 3., 1.)
    map.add_circle(3., 1., 1.)
    map.add_triangle(
        p0=[2., 2.], 
        p1=[2., 4.],
        p2=[2.5, 3.5]
    )
    map.add_triangle(
        p0=[2., 2.], 
        p1=[4., 2.],
        p2=[3.5, 2.5]
    )
    # map.add_line(2., 2., 4., 2.)
    # map.add_line(2., 2., 2., 4.)
    # map.add_line(4., 2., 2., 4.)
    app = QApplication(sys.argv)

    #window = drawwindow.TensorDisplayWindow(map, ratio=0.01)
    window = mainwindow.MainWindow()
    window.show()

    sys.exit(app.exec_())
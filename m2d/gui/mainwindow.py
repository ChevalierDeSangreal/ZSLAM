import torch
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QStackedWidget, QLineEdit, QPushButton, QMessageBox, QHBoxLayout, QMainWindow, QLabel, QVBoxLayout, QWidget, QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLUT import *
from gui.mapcreator import draw_grid, draw_new_circle, draw_new_triangle
from map import Map
from utils.geometry import not_triangle
import json


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, tensor, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.tensor = tensor
        self.texture_id = None
        self.setFixedSize(tensor.shape[0], tensor.shape[1]) 

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)
        self.update_texture()

    def update_texture(self):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        np_array = self.tensor.to(dtype=torch.uint8) * 255
        np_array = torch.stack([np_array, np_array, np_array], dim=-1)
        np_array = np_array.flip(0)
        img_data = np_array.numpy()

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_data.shape[1], img_data.shape[0], 
                     0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f( 1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f( 1.0,  1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0,  1.0)
        glEnd()

        glFlush()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.map = None
        self.ratio = None
        self.grid = None
        self.glwidget = None

        self.block_name = None
        self.block_information = None

        self.setWindowTitle("M2D")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_page = self.get_start_page()
        self.display_page = None

        self.stack.addWidget(self.start_page)
        #self.stack.addWidget(self.display_page)
        #self.stack.setCurrentIndex(0)
        self.stack.setCurrentWidget(self.start_page)
        #self.stack.setCurrentWidget(self.display_page)


    def get_start_page(self):
        page = QWidget()
        page.setFixedWidth(200)
        layout = QVBoxLayout(page)
        h_input = QLineEdit(page)
        h_input.setPlaceholderText("Height")
        w_input = QLineEdit(page)
        w_input.setPlaceholderText("Width")
        ratio_input = QLineEdit(page)
        ratio_input.setPlaceholderText("ratio")
        layout.addWidget(h_input)
        layout.addWidget(w_input)
        layout.addWidget(ratio_input)

        new_map_button = QPushButton("New Map", page)
        new_map_button.clicked.connect(
            lambda: self.new_map(h_input, w_input, ratio_input))
        layout.addWidget(new_map_button)

        load_map_button = QPushButton("Load Map", page)
        load_map_button.clicked.connect(
            lambda: self.load_map())
        layout.addWidget(load_map_button)
        layout.addStretch()
        return page

    def show_warning(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("输入错误")
        msg_box.setText("请输入有效的数字！")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def new_map(self, h_input, w_input, ratio_input):
        try:
            h, w, ratio = h_input.text(), w_input.text(), ratio_input.text()
            h_input.clear()
            w_input.clear()
            ratio_input.clear()
            h = float(h)
            w = float(w)
            ratio = float(ratio)
            if h != ratio * int(h / ratio) \
                or w != ratio * int(w / ratio) \
                    or ratio <= 0:
                self.show_warning()
                return
            map = Map(width=w, height=h, ratio=ratio)
            self.init_map(map)
            self.resize(int(map.width/map.ratio)+200, int(map.height/map.ratio))
            self.display_page = self.get_display_page()
            self.stack.addWidget(self.display_page)
            self.stack.setCurrentWidget(self.display_page)
            ###
        except ValueError:
            self.show_warning()

    def load_map(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open JSON File",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if filename:
            try:
                map = Map.from_json(filename=filename)
                self.init_map(map)
                self.resize(int(map.width/map.ratio)+200, int(map.height/map.ratio))
                self.display_page = self.get_display_page()
                self.stack.addWidget(self.display_page)
                self.stack.setCurrentWidget(self.display_page)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", "Failed to load JSON file. Please check the format.")

    def init_map(self, map:Map):
        self.map = map
        self.ratio = map.ratio
        self.grid = draw_grid(map=map, ratio=map.ratio, device="cuda:0")
        tensor = self.grid.cpu()
        self.glwidget = OpenGLWidget(tensor, self.display_page)

    def get_display_page(self):
        if self.map is not None:
            page = QWidget()
            layout = QHBoxLayout(page)
            layout.addWidget(self.glwidget)

            button_layout = QVBoxLayout()
            self.ctrl_stack = QStackedWidget(page)
            self.ctrl_stack.setFixedWidth(200)

            choose_shape = self.shape_choose_widget()
            self.ctrl_stack.addWidget(choose_shape)
            self.ctrl_stack.setCurrentWidget(choose_shape)

            button_layout.addWidget(self.ctrl_stack)
            button_layout.addStretch()
            layout.addLayout(button_layout)
            return page
        return None
    
    def shape_choose_widget(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        add_circle_button = QPushButton("Add Circle", page)
        add_triangle_button = QPushButton("Add Triangle", page)

        #add_circle_button.clicked.connect(lambda: self.add_circle())
        #add_triangle_button.clicked.connect(lambda: self.add_triangle())

        layout.addWidget(add_circle_button)
        layout.addWidget(add_triangle_button)
        return page
    
    def add_circle_widget(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        cx_input = QLineEdit(page)
        cx_input.setPlaceholderText("Center x")
        cy_input = QLineEdit(page)
        cy_input.setPlaceholderText("Center y")
        r_input = QLineEdit(page)
        r_input.setPlaceholderText("Radius")
        layout.addWidget(cx_input)
        layout.addWidget(cy_input)
        layout.addWidget(r_input)

        submit_button = QPushButton("Submit Circle", page)
        submit_button.clicked.connect(
            lambda: self.validate_input_circle(cx_input, cy_input, r_input))
        layout.addWidget(submit_button)

        return page

    def validate_input_circle(self, cx_input, cy_input, r_input):
        try:
            cx, cy, r = cx_input.text(), cy_input.text(), r_input.text()
            cx_input.clear()
            cy_input.clear()
            r_input.clear()
            cx = float(cx)
            cy = float(cy)
            r = float(r)
            if cx != self.ratio * int(cx / self.ratio) \
                or cy != self.ratio * int(cy / self.ratio) \
                    or r != self.ratio * int(r / self.ratio) \
                        or r <= 0:
                self.show_warning()
                return
            self.add_cirile(x=cx, y=cy, r=r)
            self.block_name = "circle"
            self.block_information = [cx, cy, r]
            self.ctrl_stack.setCurrentWidget(self.change_map)

        except ValueError:
            self.show_warning()
            
    def add_cirile(self, x, y, r):
        grid = draw_new_circle(self.grid, x=x, y=y, r=r, ratio=self.ratio)
        self.glwidget.tensor = grid.cpu()
        self.glwidget.update_texture()

    def add_triangle_widget(self):
        page = QWidget()

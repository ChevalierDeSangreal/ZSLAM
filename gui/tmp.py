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

        # 上传到 OpenGL 纹理
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_data.shape[1], img_data.shape[0], 
                     0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

        # 设置纹理参数
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

class TensorDisplayWidget(QWidget):
    def __init__(self, map:Map, ratio=0.01, parent=None):
        super().__init__(parent)
        self.init_map(map=map, ratio=ratio)

        self.block_name = None
        self.block_information = None

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.glwidget)

        button_layout = QVBoxLayout()
        self.ctrl_stack = QStackedWidget()
        self.ctrl_stack.setFixedWidth(200)

        self.get_param = self.param_widget()
        self.change_map = self.save_or_cancel()

        self.ctrl_stack.addWidget(self.get_param)
        self.ctrl_stack.addWidget(self.change_map)

        button_layout.addWidget(self.ctrl_stack)
        button_layout.addStretch() 

        self.layout.addLayout(button_layout)

    def init_map(self, map:Map, ratio=0.01):
        self.map = map
        self.ratio = ratio
        self.grid = draw_grid(map=map, ratio=ratio, device="cuda:0")
        tensor = self.grid.cpu()
        self.glwidget = OpenGLWidget(tensor, self)

    def save_or_cancel(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        save_button = QPushButton("Save", page)
        save_button.clicked.connect(lambda: self.save_to_map(self.block_name, self.block_information))
        layout.addWidget(save_button)

        cancel_button = QPushButton("Cancel", page)
        cancel_button.clicked.connect(lambda: self.cancel())
        layout.addWidget(cancel_button)
        
        return page

    def save_to_map(self, block_name, block_information):
        if block_name == "circle":
            self.map.add_circle(
                x=block_information[0],
                y=block_information[1],
                r=block_information[2]
            )
            self.grid = self.glwidget.tensor.to("cuda:0")
            self.ctrl_stack.setCurrentWidget(self.get_param)
            self.glwidget.update_texture()
        
        elif block_name == "triangle":
            self.map.add_triangle(
                p0=block_information[0],
                p1=block_information[1],
                p2=block_information[2]
            )
            self.grid = self.glwidget.tensor.to("cuda:0")
            self.ctrl_stack.setCurrentWidget(self.get_param)
            self.glwidget.update_texture()
        
        self.block_name = None
        self.block_information = None

    def cancel(self):
        self.glwidget.tensor = self.grid.cpu()
        self.ctrl_stack.setCurrentWidget(self.get_param)
        self.glwidget.update_texture()

    def add_cirile(self, x, y, r):
        grid = draw_new_circle(self.grid, x=x, y=y, r=r, ratio=self.ratio)
        self.glwidget.tensor = grid.cpu()
        self.glwidget.update_texture()

    def add_triangle(self, p0, p1, p2):
        grid = draw_new_triangle(self.grid, p0, p1, p2, ratio=self.ratio)
        self.glwidget.tensor = grid.cpu()
        self.glwidget.update_texture()

    def param_widget(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        cx_input = QLineEdit(page)
        cx_input.setPlaceholderText("center x")
        cy_input = QLineEdit(page)
        cy_input.setPlaceholderText("center y")
        r_input = QLineEdit(page)
        r_input.setPlaceholderText("radius")
        layout.addWidget(cx_input)
        layout.addWidget(cy_input)
        layout.addWidget(r_input)

        submit_circle_button = QPushButton("submit circle", page)
        submit_circle_button.clicked.connect(lambda: self.validate_input_circle(cx_input, cy_input, r_input))
        layout.addWidget(submit_circle_button)

        x0_input = QLineEdit(page)
        x0_input.setPlaceholderText("x0")
        y0_input = QLineEdit(page)
        y0_input.setPlaceholderText("y0")
        x1_input = QLineEdit(page)
        x1_input.setPlaceholderText("x1")
        y1_input = QLineEdit(page)
        y1_input.setPlaceholderText("y1")
        x2_input = QLineEdit(page)
        x2_input.setPlaceholderText("x2")
        y2_input = QLineEdit(page)
        y2_input.setPlaceholderText("y2")

        layout.addWidget(x0_input)
        layout.addWidget(y0_input)
        layout.addWidget(x1_input)
        layout.addWidget(y1_input)
        layout.addWidget(x2_input)
        layout.addWidget(y2_input)

        submit_triangle_button = QPushButton("submit triangle", page)
        submit_triangle_button.clicked.connect(lambda: self.validate_input_triangle(x0_input, y0_input, x1_input, y1_input, x2_input, y2_input))
        layout.addWidget(submit_triangle_button)

        save_map_button = QPushButton("save map file", page)
        save_map_button.clicked.connect(self.save_map_file)
        layout.addWidget(save_map_button)

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

    
    def validate_input_triangle(self, x0_input, y0_input, x1_input, y1_input, x2_input, y2_input):
        try:
            x0, y0, x1, y1, x2, y2 = x0_input.text(), y0_input.text(), x1_input.text(), y1_input.text(), x2_input.text(), y2_input.text()
            x0_input.clear()
            y0_input.clear()
            x1_input.clear()
            y1_input.clear()
            x2_input.clear()
            y2_input.clear()
            x0, y0, x1, y1, x2, y2 = float(x0), float(y0), float(x1), float(y1), float(x2), float(y2)
            if x0 != self.ratio * int(x0 / self.ratio) \
                or y0 != self.ratio * int(y0 / self.ratio) \
                    or x1 != self.ratio * int(x1 / self.ratio) \
                        or y1 != self.ratio * int(y1 / self.ratio) \
                            or x2 != self.ratio * int(x2 / self.ratio) \
                                or y2 != self.ratio * int(y2 / self.ratio) \
                                    or not_triangle([x0, y0], [x1, y1], [x2, y2]):
                self.show_warning()
                return
            self.add_triangle([x0, y0], [x1, y1], [x2, y2])
            self.block_name = "triangle"
            self.block_information = [[x0, y0], [x1, y1], [x2, y2]]
            self.ctrl_stack.setCurrentWidget(self.change_map)

        except ValueError:
            self.show_warning()
            
    def show_warning(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("输入错误")
        msg_box.setText("请输入有效的数字！")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def save_map_file(self):
        json_data = self.map.to_json_data()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "SAVE JSON FILE", 
            "data.json",
            "JSON Files (*.json);;All Files (*)",
            options=options)
        
        if file_path:
            if not file_path.endswith(".json"):
                file_path += ".json"
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M2D")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_page = self.get_start_page()
        self.display_page = TensorDisplayWidget(
            map=Map(1., 1., 1.), 
            parent=self
        )

        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.display_page)

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
            self.map = Map(width=w, height=h, ratio=ratio)
            print(self.map.to_json_data())

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
                self.resize(int(map.width/map.ratio)+200, int(map.height/map.ratio))
                self.display_page.init_map(map=map, ratio=map.ratio)
                #self.stack.addWidget()
                self.stack.setCurrentWidget(self.display_page)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", "Failed to load JSON file. Please check the format.")
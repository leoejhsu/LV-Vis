import pandas as pd
import ast
from vispy.visuals.transforms import STTransform
from vispy import scene, app
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
import numpy as np
import os

from .my_vispy.my_volume import MyVisual



class AxisVisualizer:
    def __init__(self, view, camera):
        self.axis = scene.visuals.XYZAxis(parent=view, width=10.0)
        self.axis.set_gl_state(blend=False)
        self.axis.transform = STTransform(translate=(100, 100), scale=(100, 100, 100, 1)).as_matrix()

        self.camera = camera
        self.camera.transform.changed.connect(lambda event: self.update_axis_visual())

    def update_axis_visual(self):
        """Sync XYZAxis visual with camera angles"""
        self.axis.transform.reset()
        self.axis.transform.rotate(self.camera.roll, (0, 0, 1))
        self.axis.transform.rotate(self.camera.elevation, (1, 0, 0))
        self.axis.transform.rotate(self.camera.azimuth, (0, 1, 0))
        self.axis.transform.scale((100, 100, 0.001))
        self.axis.transform.translate((100., 100.))

        self.axis.update()


class CameraController:
    def __init__(self, view):
        self.camera = scene.cameras.TurntableCamera(parent=view.scene, fov=0)
        self.camera.scale_factor = 500.0
        # self.camera.center = np.array([80., 80., 67.5]) + np.array([0., 0., 127.])
        # self.camera.azimuth = 0.0
        # self.camera.elevation = 0.0
        # self.camera.roll = 0.0
        view.camera = self.camera


class CanvasManager:
    def __init__(self, parent_widget, w, h):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(w, h), show=True)
        self.view = self.canvas.central_widget.add_view()
        # layout = QHBoxLayout(parent_widget)
        # layout.addWidget(self.canvas.native)
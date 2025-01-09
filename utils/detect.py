import cv2
from PIL import Image
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QPixmap
from .tracker import Tracker



class Detect(QThread):
    detected = Signal(QPixmap)

    def __init__(self) -> None:
        super().__init__()
        self.video = cv2.VideoCapture('resource/zzzz.mp4')
        self.traker = Tracker([320, 320])
        self._pause = False
        if not self.video.isOpened():
            raise ValueError('open video failed !')
        
    def run(self):
        self._pause = False

        while not self._pause:
            ret, frame = self.video.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plotted = self.traker(frame)
            plotted = Image.fromarray(plotted).toqpixmap()
            self.detected.emit(plotted)

    def pause(self):
        self._pause = True

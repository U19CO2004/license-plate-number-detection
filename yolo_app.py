from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np
import sys

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize YOLO model
        self.model = YOLO("best.pt")  # Replace with your YOLO model path

        # Set up the UI
        self.setWindowTitle("YOLOv8 Image Prediction")
        self.setGeometry(200, 200, 800, 600)

        # Layout and widgets
        self.layout = QVBoxLayout()

        # QLabel for displaying images
        self.image_label = QLabel("Load an image to see predictions")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # QPushButton to load an image
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        # Set the layout
        self.setLayout(self.layout)

    def load_image(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)

        if file_path:
            # Run YOLO inference
            results = self.model.predict(source=file_path, save=True, save_txt=True)
            result = results[0]  # Get the first result

            # Annotate the image
            annotated_img = result.plot()

            # Convert the image to QImage format for PyQt5
            height, width, channel = annotated_img.shape
            bytes_per_line = channel * width
            qimg = QImage(annotated_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Display the image in the QLabel
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)  # Scale the image to fit the label size


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())

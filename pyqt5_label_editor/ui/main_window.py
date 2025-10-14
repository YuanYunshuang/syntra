import os

import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPolygon, QColor
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_index = 0
        self.img_files = []
        self.lbl_files = []
        self.img_folder = ""
        self.lbl_folder = ""
        self.image_label = QLabel(self)
        self.label_label = QLabel(self)
        self.selected_color = None
        self.selected_area = None
        self.drawing_mode = False

        # Set fixed size for image and label display
        self.image_label.setFixedSize(600, 600)
        self.label_label.setFixedSize(600, 600)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        display_layout = QHBoxLayout()  # Horizontal layout for image and label
        self.legend_layout = QVBoxLayout()

        load_button = QPushButton("Load Dataset", self)
        load_button.clicked.connect(self.load_images)

        prev_button = QPushButton("Previous", self)
        prev_button.clicked.connect(self.show_previous_image)

        next_button = QPushButton("Next", self)
        next_button.clicked.connect(self.show_next_image)

        remove_button = QPushButton("Remove", self)
        remove_button.clicked.connect(self.remove_selected_area)

        button_layout.addWidget(load_button)
        button_layout.addWidget(prev_button)
        button_layout.addWidget(next_button)
        button_layout.addWidget(remove_button)

        # Add image and label widgets to the horizontal layout
        display_layout.addWidget(self.image_label)
        display_layout.addWidget(self.label_label)
        display_layout.addLayout(self.legend_layout)

        layout.addLayout(button_layout)
        layout.addLayout(display_layout)  # Add the horizontal layout to the main layout

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # show the winqdow
        self.setWindowTitle("Image and Label Viewer")
        self.resize(1200, 650)

    def load_images(self):
        root = QFileDialog.getExistingDirectory(self, "Select Data Folder", "/home/yuan/data/HisMap/syntra384_correction")
        img_folder = os.path.join(root, 'imgs')
        lbl_folder = os.path.join(root, 'lbls')

        if img_folder and lbl_folder:
            self.img_files = sorted(os.listdir(img_folder))
            self.lbl_files = sorted(os.listdir(lbl_folder))
            self.img_folder = img_folder
            self.lbl_folder = lbl_folder
            self.image_index = 0
            self.display_current_image()

    def display_current_image(self):
        if self.img_files and self.lbl_files:
            img_path = os.path.join(self.img_folder, self.img_files[self.image_index])
            lbl_path = os.path.join(self.lbl_folder, self.lbl_files[self.image_index])
            self.show_image(img_path)
            self.show_label(lbl_path)

    def show_image(self, img_path):
        pixmap = QPixmap(img_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def show_label(self, lbl_path):
        image = QImage(lbl_path)
        self.label_label.setPixmap(QPixmap.fromImage(image).scaled(self.label_label.size(), Qt.KeepAspectRatio))

    def show_next_image(self):
        if self.image_index < len(self.img_files) - 1:
            self.image_index += 1
            self.display_current_image()

    def show_previous_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.display_current_image()

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.current_polygon:
            pos = event.pos() - self.image_label.pos()
            x = int(pos.x() * self.image_label.pixmap().width() / self.image_label.width())
            y = int(pos.y() * self.image_label.pixmap().height() / self.image_label.height())
            if self.current_polygon:
                last_point = self.current_polygon[-1]
                self.draw_dynamic_line(last_point, (x, y))

    def draw_dynamic_line(self, start_point, end_point):
        # Create a transparent overlay
        overlay = QPixmap(self.image_label.size())
        overlay.fill(Qt.transparent)

        painter = QPainter(overlay)
        painter.setPen(QColor(255, 0, 0, 128))  # Semi-transparent red
        pen = painter.pen()
        pen.setStyle(Qt.DashLine)  # Dashed line
        painter.setPen(pen)

        # Draw the dynamic line
        painter.drawLine(QPoint(*start_point), QPoint(*end_point))
        painter.end()

        # Combine the overlay with the original image
        base_pixmap = self.image_label.pixmap()
        combined_pixmap = QPixmap(base_pixmap.size())
        combined_pixmap.fill(Qt.transparent)

        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, base_pixmap)
        painter.drawPixmap(0, 0, overlay)
        painter.end()

        # Set the combined pixmap back to the image_label
        self.image_label.setPixmap(combined_pixmap)

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton and self.image_label.underMouse():
            pos = event.pos() - self.image_label.pos()
            x = int(pos.x() * self.image_label.pixmap().width() / self.image_label.width())
            y = int(pos.y() * self.image_label.pixmap().height() / self.image_label.height())
            # draw the point
            if not self.current_polygon:
                self.draw_polygon_point(x, y)
            else:
                self.draw_dynamic_polygon(x, y)
            self.current_polygon.append((x, y))  # Add the point to the polygon
        if event.button() == Qt.LeftButton and self.label_label.underMouse():
            pos = event.pos() - self.label_label.pos()
            x = int(pos.x() * self.label_label.pixmap().width() / self.label_label.width())
            y = int(pos.y() * self.label_label.pixmap().height() / self.label_label.height())

            label_image = self.label_label.pixmap().toImage()
            color = label_image.pixelColor(x, y)
            self.selected_color = color

            # Add a button with the same color
            self.add_legend_button(color)

            # Select the coherent area
            self.selected_area = self.flood_fill(label_image, x, y, color) # a np 2d mask

            # Overlay the mask on the label image
            self.overlay_mask_on_label(self.selected_area, color)
    
    def draw_polygon_point(self, x, y):
        # Create a painter to draw directly on the image label's pixmap
        pixmap = self.image_label.pixmap()
        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 0, 0))  # Red color for the point
        painter.setPen(Qt.NoPen)

        # Draw a small square centered at (x, y)
        point_size = 5
        painter.drawRect(x - point_size // 2, y - point_size // 2, point_size, point_size)
        painter.end()

        # Update the image label to reflect the changes
        self.image_label.setPixmap(pixmap)
    
    def overlay_mask_on_label(self, mask, color):
        # Convert the mask to an RGBA image
        height, width = mask.shape
        overlay = np.zeros((height, width, 4), dtype=np.uint8)  # Create an RGBA image
        overlay[mask] = [color.red(), color.green(), color.blue(), 128]  # Apply the selected color with 50% transparency

        # Convert the overlay to a QImage
        overlay_image = QImage(overlay.data, width, height, QImage.Format_ARGB32)

        # Combine the overlay with the original label image
        image_pixmap = self.image_label.pixmap()
        combined_pixmap = QPixmap(self.image_label.size())
        combined_pixmap.fill(Qt.transparent)

        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, image_pixmap)  # Draw the original label image
        painter.drawImage(0, 0, overlay_image)  # Draw the overlay on top
        painter.end()

        # Set the combined pixmap back to the label
        self.image_label.setPixmap(combined_pixmap)


    def add_legend_button(self, color):
        button = QPushButton(self)
        button.setStyleSheet(f"background-color: {color.name()}")
        button.clicked.connect(lambda: self.activate_drawing_mode(color))  # Activate drawing mode on click
        self.legend_layout.addWidget(button)

    def activate_drawing_mode(self, color):
        self.drawing_mode = True
        self.drawing_color = color
        self.current_polygon = []  # Store the points of the polygon

    def flood_fill(self, image, x, y, target_color):
        width, height = image.width(), image.height()
        visited = np.zeros((height, width), dtype=bool)
        stack = [(x, y)]
        area = []

        while stack:
            cx, cy = stack.pop()
            if not (0 <= cx < width and 0 <= cy < height):
                continue
            if visited[cy, cx]:
                continue
            if image.pixelColor(cx, cy) != target_color:
                continue

            visited[cy, cx] = True
            area.append((cx, cy))
            stack.extend([[cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]])
        
        # convert area to a 2d mask
        area_mask = np.zeros((height, width), dtype=bool)
        xys = np.array(area)
        area_mask[xys[:, 1], xys[:, 0]] = True
        return area_mask

    def remove_selected_area(self):
        if self.selected_area:
            label_image = Image.open(self.current_label_path)
            label_array = np.array(label_image)

            for x, y in self.selected_area:
                label_array[y, x] = 0  # Set the pixel to black (or any other background color)

            updated_label = Image.fromarray(label_array)
            updated_label.save(self.current_label_path)

            self.show_label(self.current_label_path)  # Refresh the label display
    
    def draw_dynamic_polygon(self, x, y):
        # Create a temporary polygon with the current points and the dynamic point
        temp_polygon = self.current_polygon + [(x, y)]

        # Create a transparent overlay
        overlay = QPixmap(self.image_label.size())
        overlay.fill(Qt.transparent)

        painter = QPainter(overlay)

        # Draw the previous edges as solid red lines
        if len(self.current_polygon) > 1:
            painter.setPen(QColor(255, 0, 0))  # Solid red line
            for i in range(len(self.current_polygon) - 1):
                start_point = QPoint(*self.current_polygon[i])
                end_point = QPoint(*self.current_polygon[i + 1])
                painter.drawLine(start_point, end_point)

        # Draw the current edge as a dashed red line
        if self.current_polygon:
            painter.setPen(QColor(255, 0, 0, 128))  # Semi-transparent red
            pen = painter.pen()
            pen.setStyle(Qt.DashLine)  # Dashed line
            painter.setPen(pen)
            last_point = QPoint(*self.current_polygon[-1])
            dynamic_point = QPoint(*temp_polygon[-1])
            painter.drawLine(last_point, dynamic_point)

        painter.end()

        # Combine the overlay with the original image
        base_pixmap = self.image_label.pixmap()
        combined_pixmap = QPixmap(base_pixmap.size())
        combined_pixmap.fill(Qt.transparent)

        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, base_pixmap)
        painter.drawPixmap(0, 0, overlay)
        painter.end()

        # Set the combined pixmap back to the image_label
        self.image_label.setPixmap(combined_pixmap)

    def keyPressEvent(self, event):
        if self.drawing_mode and event.key() == Qt.Key_D:  # Finish the polygon on 'd' key press
            self.finish_polygon()

    def finish_polygon(self):
        # Create a final overlay with the selected color
        overlay = QPixmap(self.image_label.size())
        overlay.fill(Qt.transparent)

        painter = QPainter(overlay)
        painter.setBrush(QColor(self.drawing_color.red(), self.drawing_color.green(), self.drawing_color.blue(), 128))  # Semi-transparent selected color
        painter.setPen(Qt.NoPen)

        # Draw the final polygon
        qpolygon = QPolygon([QPoint(px, py) for px, py in self.current_polygon])
        painter.drawPolygon(qpolygon)
        painter.end()

        # Combine the overlay with the original image
        base_pixmap = self.image_label.pixmap()
        combined_pixmap = QPixmap(base_pixmap.size())
        combined_pixmap.fill(Qt.transparent)

        painter = QPainter(combined_pixmap)
        painter.drawPixmap(0, 0, base_pixmap)
        painter.drawPixmap(0, 0, overlay)
        painter.end()

        # Set the combined pixmap back to the image_label
        self.image_label.setPixmap(combined_pixmap)

        # Reset drawing mode
        self.drawing_mode = False
        self.current_polygon = []
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsScene, QGraphicsView, QGraphicsPolygonItem
from PyQt5.QtGui import QImage, QPainter, QPolygonF, QColor
from PyQt5.QtCore import Qt, QPointF
import os

class LabelEditor(QWidget):
    def __init__(self, img_folder, lbl_folder):
        super().__init__()
        self.img_folder = img_folder
        self.lbl_folder = lbl_folder
        self.image = None
        self.label = None
        self.selected_class_color = QColor(255, 0, 0)  # Default to red for drawing
        self.drawing_mode = False
        self.segments = []
        
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.layout.addWidget(self.graphics_view)

        self.setLayout(self.layout)

    def load_image(self, image_name):
        image_path = os.path.join(self.img_folder, image_name)
        self.image = QImage(image_path)
        self.update_display()

    def load_label(self, label_name):
        label_path = os.path.join(self.lbl_folder, label_name)
        self.label = QImage(label_path)
        self.update_display()

    def update_display(self):
        if self.image and self.label:
            combined_image = QImage(self.image.size(), QImage.Format_ARGB32)
            painter = QPainter(combined_image)
            painter.drawImage(0, 0, self.image)
            painter.drawImage(0, 0, self.label)
            painter.end()
            self.image_label.setPixmap(QPixmap.fromImage(combined_image))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.label:
            pos = event.pos()
            # Logic to select and highlight segment based on pixel
            self.highlight_segment(pos)

    def highlight_segment(self, pos):
        # Implement logic to highlight the segment at the clicked position
        pass

    def delete_segment(self):
        # Implement logic to delete the selected segment
        pass

    def activate_drawing_mode(self):
        self.drawing_mode = True

    def draw_polygon(self, points):
        if self.drawing_mode:
            polygon = QPolygonF([QPointF(p[0], p[1]) for p in points])
            polygon_item = QGraphicsPolygonItem(polygon)
            polygon_item.setBrush(self.selected_class_color)
            self.scene.addItem(polygon_item)

    def set_selected_class_color(self, color):
        self.selected_class_color = color
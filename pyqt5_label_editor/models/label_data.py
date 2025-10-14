class LabelData:
    def __init__(self):
        self.images = []
        self.labels = []
        self.palette = {}

    def load_data(self, img_folder, lbl_folder):
        self.images = self._load_images(img_folder)
        self.labels = self._load_labels(lbl_folder)

    def _load_images(self, folder):
        # Logic to load images from the specified folder
        pass

    def _load_labels(self, folder):
        # Logic to load label files from the specified folder
        pass

    def set_palette(self, palette):
        self.palette = palette

    def get_palette(self):
        return self.palette

    def get_image(self, index):
        return self.images[index] if index < len(self.images) else None

    def get_label(self, index):
        return self.labels[index] if index < len(self.labels) else None

    def delete_segment(self, index):
        if 0 <= index < len(self.labels):
            self.labels.pop(index)
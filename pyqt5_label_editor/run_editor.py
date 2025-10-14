import sys
import os

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def copy_dataset(src, dst, split_name):
    import shutil
    # create dst folder
    img_folder = os.path.join(dst, 'imgs')
    lbl_folder = os.path.join(dst, 'lbls')
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(lbl_folder, exist_ok=True)

    # get sample names from src
    if 'nshot' in split_name:
        nshot = int(split_name.split('nshot')[-1])
        # load json file
        import json
        with open(os.path.join(src, f'selected_{nshot}.json'), 'r') as f:
            data_info = json.load(f)
            all_samples = []
        for cls_samples in data_info.values():
            all_samples += cls_samples
        all_samples_no_repeat = list(set(all_samples))
    else:
        with open(os.path.join(src, f'{split_name}_samples.txt'), 'r') as f:
            all_samples = f.read().splitlines()
        all_samples_no_repeat = all_samples

    print(f"Copying {len(all_samples_no_repeat)} samples from {src} to {dst} ...")
    
    for sample in all_samples_no_repeat:
        sample_name = sample.rsplit('.', 1)[1] + '.png'
        shutil.copy(os.path.join(src, 'imgs', sample_name), os.path.join(img_folder, sample_name))
        shutil.copy(os.path.join(src, 'lbls', sample_name), os.path.join(lbl_folder, sample_name))

if __name__ == "__main__":
    main()
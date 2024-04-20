import os
import sys
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from utils.data_RGB import get_test_data, get_test_data_img
from models.nerd_s import MultiscaleNet
from skimage import img_as_ubyte
from utils.get_parameter_number import get_parameter_number
from tqdm import tqdm
from models.layers import *
from plate import *
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap, QImage
from main_ui import Ui_MainWindow


parser = argparse.ArgumentParser(description='Image Deraining')
parser.add_argument('--input_dir', default='imgs/input/rain_effct_result.jpg', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='imgs/output', type=str, help='Directory of validation images')
parser.add_argument('--nerd_weights', default='weights/nerd_self_s.pth', type=str, help='Path to weights') 
parser.add_argument('--plate_weights', nargs='+', type=str, default='weights/plate_detect.pt', help='model.pt path(s)')  #检测模型
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--win_size', default=256, type=int, help='window size')
parser.add_argument('--rec_model', type=str, default='weights/plate_rec.pth', help='model.pt path(s)')#车牌识别+颜色识别模型
parser.add_argument('--is_color',type=bool,default=True,help='plate color')      #是否识别颜色
parser.add_argument('--image_path', type=str, default='imgs', help='source')     #图片路径
parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  #网络输入图片大小
parser.add_argument('--output', type=str, default='result', help='source')               #图片结果保存的位置
parser.add_argument('--video', type=str, default='', help='source')                       #视频的路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
result_dir = args.output_dir
win = args.win_size
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def convert2QImage(image):
    height, width, channel = image.shape
    return QImage(image, width, height, width * channel, QImage.Format_BGR888)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.bind_shots()
        # nerd
        self.model_restoration = MultiscaleNet()
        # get_parameter_number(self.model_restoration)
        utils.load_checkpoint(self.model_restoration, args.nerd_weights)
        self.model_restoration.cuda()
        self.model_restoration = nn.DataParallel(self.model_restoration)
        self.model_restoration.eval()
        
        # plate
        self.detect_model = load_model(args.plate_weights, device)  #初始化检测模型
        self.plate_rec_model = init_model(device, args.rec_model, is_color = args.is_color)

        # dataset
        rgb_dir_test = args.input_dir
        test_dataset = get_test_data_img(rgb_dir_test, img_options={})
        self.test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    def bind_shots(self):
        self.btn_input.clicked.connect(self.load_img)
        self.btn_detect.clicked.connect(self.detect)
        self.closeAppBtn.clicked.connect(self.close)
        self.btn_save.clicked.connect(self.save)
        self.btn_exit.clicked.connect(self.clear)

    def detect_img(self, filt_path):
        rgb_dir_test = filt_path
        test_dataset = get_test_data_img(rgb_dir_test, img_options={})
        test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            for ii, data_test in enumerate(tqdm(test_loader), 0):

                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                input_    = data_test[0].cuda()
                filenames = data_test[1]
                _, _, Hx, Wx = input_.shape
                filenames = data_test[1]
                input_re, batch_list = window_partitionx(input_, win)
                restored = self.model_restoration(input_re)
                restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

                restored = torch.clamp(restored, 0, 1)
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(restored)):
                    restored_img = restored[batch]
                    restored_img = img_as_ubyte(restored[batch])
                    
                    # utils.save_img(result_dir + '/result'+'.png', restored_img)

        # plate
        restored = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        dict_list = detect_Recognition_plate(self.detect_model, restored, device, self.plate_rec_model, args.img_size, is_color=args.is_color)#检测以及识别车牌
        restored = np.ascontiguousarray(restored)
        ori_img, result_str = draw_result(restored, dict_list)

        return cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR), ori_img, result_str

    def load_img(self):
        filt_path = QFileDialog.getOpenFileName(self, dir="./", filter="*.png; *.jpg; *.jpeg")
        if filt_path[0]:
            self.info.setText('已读取图像')
            self.filt_path = filt_path[0]
            self.input.setPixmap(QPixmap(self.filt_path))
            self.input.setScaledContents (True)

    def detect(self):
        self.restored_img, self.ori_img, result_str = self.detect_img(self.filt_path)
        restored_img, ori_img = convert2QImage(self.restored_img), convert2QImage(self.ori_img)
        self.output1.setPixmap(QPixmap.fromImage(restored_img.scaled(self.output1.width(), self.output1.height())))
        self.output2.setPixmap(QPixmap.fromImage(ori_img.scaled(self.output2.width(), self.output2.height())))
        self.output1.setScaledContents (True)
        self.output2.setScaledContents (True)
        self.output_text.setStyleSheet("background:rgb(65, 71, 85); font-size: 16px;")
        self.output_text.setText('\t'+result_str)
        self.info.setText('检测完毕!')

    def save(self):
        if self.output2:
            file_name, _ = QFileDialog.getSaveFileName(self, "save_result", "./", "Image Files (*.png *.jpg *.bmp)")
            if file_name:
                cv2.imwrite(file_name, self.ori_img)
                self.info.setText(f'已保存至{file_name}')

    def clear(self):
        self.input.setPixmap(QPixmap())
        self.output1.setPixmap(QPixmap())
        self.output2.setPixmap(QPixmap())
        self.info.setText('')
        self.output_text.setText('')

    def close(self):
        App = QApplication.instance()
        App.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
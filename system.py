import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from utils.data_RGB import get_test_data, get_test_data_img
from models.nerd_s import MultiscaleNet
# from models.rend import MultiscaleNet
from skimage import img_as_ubyte
from utils.get_parameter_number import get_parameter_number
from tqdm import tqdm
from models.layers import *
from plate import *

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

##  model init
# nerd
model_restoration = MultiscaleNet()
get_parameter_number(model_restoration)
utils.load_checkpoint(model_restoration, args.nerd_weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
# plate
detect_model = load_model(args.plate_weights, device)  #初始化检测模型
plate_rec_model = init_model(device, args.rec_model, is_color = args.is_color)
# print("===>Testing using weights: ",args.weights)

# dataset = args.dataset
rgb_dir_test = args.input_dir
test_dataset = get_test_data_img(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)


utils.mkdir(result_dir)

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
        restored = model_restoration(input_re)
        restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img(result_dir + '/result'+'.png', restored_img)

# plate
restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

total = sum(p.numel() for p in detect_model.parameters())
total_1 = sum(p.numel() for p in plate_rec_model.parameters())

time_all = 0
time_begin = time.time()
time_b = time.time()

dict_list = detect_Recognition_plate(detect_model, restored_img, device, plate_rec_model, args.img_size, is_color=args.is_color)#检测以及识别车牌
restored_img = np.ascontiguousarray(restored_img)
ori_img, result_str = draw_result(restored_img, dict_list)

print(result_str)
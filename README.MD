# 恶劣天气下的车牌识别系统

![demo](./demo/demo.jpg)
## Install
```bash
pip install -r requirements.txt
```

## Usage
### 权重文件下载
| Models | S | L |
|:-----: | :-----: | :-----: |
| 去雨 | [Baidu Netdisk](https://pan.baidu.com/s/1BuJpzvzLlyN0iZQXMB6kfQ?pwd=6bsa) (6bsa) | [Baidu Netdisk](https://pan.baidu.com/s/1BuJpzvzLlyN0iZQXMB6kfQ?pwd=6bsa) (6bsa) |

> 也可以使用 [NeRD-Rain](https://github.com/cschenxiang/NeRD-Rain.git) 中的权重文件

| Models | plate_detect.pt | plate_rec.pth |
|:-----: |:-----: | :-----: |
| 车牌识别 |  [Baidu Netdisk](https://pan.baidu.com/s/1BuJpzvzLlyN0iZQXMB6kfQ?pwd=6bsa) (6bsa)  |  [Baidu Netdisk](https://pan.baidu.com/s/1BuJpzvzLlyN0iZQXMB6kfQ?pwd=6bsa) (6bsa)  |

下载后放置于 `weights/` 中，使其看起来像
```
├───demo
├───fonts
├───images
├───models
├───plate_recognition
├───result
├───utils
├───weights
│   ├───nerd_self_l.pth
│   ├───nerd_self_s.pth
│   ├───plate_detect.pt
│   ├───plate_rec.pth
│───...
```

### 运行
```bash
python main.py
```

## Citation
本工程参考如下工作

```
@InProceedings{NeRD-Rain,
    author={Chen, Xiang and Pan, Jinshan and Dong, Jiangxin}, 
    title={Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2024},
}
url = {https://github.com/cschenxiang/NeRD-Rain.git}
```

```
@software{yolov5,
  title = {Ultralytics YOLOv5},
  author = {Glenn Jocher},
  year = {2020},
  version = {7.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/yolov5},
  doi = {10.5281/zenodo.3908559},
  orcid = {0000-0001-5950-6979}
}
```

```
url = {https://github.com/we0091234/Chinese_license_plate_detection_recognition.git}
```
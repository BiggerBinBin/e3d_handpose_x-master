# Easy 3D HandPose X  
Easy 3D HandPose，pytorch，单目相机的手三维姿态估计

## 项目介绍   
* 注意：该项目前向推理用到项目包括：  
- [x] 手检测：https://codechina.csdn.net/EricLee/yolo_v3
- [x] 手二维关键点检测：https://codechina.csdn.net/EricLee/handpose_x
- [x] Manopth：https://github.com/hassony2/manopth
- [x] Easy 3D HandPose X：https://codechina.csdn.net/EricLee/e3d_handpose_x

* 视频示例：  
 ![video](https://codechina.csdn.net/EricLee/e3d_handpose_x/-/raw/master/samples/sample.gif)    
* [完整示例视频](https://www.bilibili.com/video/BV1j64y1r7da/)       
## 项目推理 PipeLine
 ![pipeline](https://codechina.csdn.net/EricLee/e3d_handpose_x/-/raw/master/samples/e3d.jpg)  
## 项目配置  
### 1、软件  
* 作者开发环境：  
* Python 3.7  
* PyTorch >= 1.5.1  
* opencv-python  
* open3d  
### 2、硬件  
* 普通USB彩色（RGB）网络摄像头    

## 数据集   
* 制作数据集，后续更新

## 模型   
### 1、目前支持的模型 (backbone)

- [x] resnet18 & resnet34 & resnet50 & resnet101

### 2、预训练模型   

* [预训练模型下载地址(百度网盘 Password: 95t4 )](https://pan.baidu.com/s/1L9JVjnvKDjG0opIAUZOF0g)        


## 项目使用方法  
### [1]准备左右手3D建模资源（MANO Hand Model）  
* 该模型也可在官网下载，官网地址为：https://mano.is.tue.mpg.de/
* 下载模型和文件 (下载文件的格式为 mano_v*_*.zip)。注意这些文件的下载使用遵守 [MANO license](https://mano.is.tue.mpg.de/license)。
* 下载模型后进行解压，并将其目录结构设置如下：
```
e3d_handpose_x/
  mano/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
      ...
    webuser/
      ...
    __init__.py
```
### [2]模型训练  
#### 根目录下运行命令： python train.py       (注意脚本内相关参数配置 )   

### [3]模型推理  
#### 根目录下运行命令：
#### 1) 图片推理脚本 ：python inference.py        (注意脚本内相关参数配置 )

#### 2) 相机推理脚本 ： python yolo_inference.py        (注意脚本内相关参数配置 )  
####    注意：目前推理为 “预发版本”，只支持画面中出现一只手会进行三维姿态估计，并只支持右手姿态估计。   

#####    第 1 步：确定电脑连接相机。
#####    第 2 步：下载 [模型前向推理包(百度网盘 Password: xhd3 )](https://pan.baidu.com/s/1wqhIgciL5mnlT1PyHKI6QQ)    
#####    第 3 步：解压模型前向推理包，配置 [yolo_inference.py](https://codechina.csdn.net/EricLee/e3d_handpose_x/-/blob/master/yolo_inference.py)脚本模型路径参数，参考如下：  
```
  parser.add_argument('--model_path', type=str, default = './if_package/e3d_handposex-resnet_50-size-128-loss-wing_loss-20210619.pth',
      help = 'model_path') # e3d handpose 模型路径
  parser.add_argument('--detect_model_path', type=str, default = './if_package/hand_detect_416-20210606.pt',
      help = 'model_path') # detect 模型路径
  parser.add_argument('--handpose_x2d_model_path', type=str, default = './if_package/handposex_2d_resnet_50-size-256-wingloss102-0.119.pth',
      help = 'model_path') # 手2维关键点 模型路径
```
#####   第 4 步：运行脚本：python yolo_inference.py
####   注意：运行出错，注意看log报的错误，尽量自行解决，思考尝试解决不了，issue提问。

## 联系方式 （Contact）  
* E-mails: 305141918@qq.com   

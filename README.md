# **说明**
这个项目是克隆到Eric.Lee2021在code china上的一个项目。
原项目地址：[https://codechina.csdn.net/EricLee/e3d_handpose_x](https://codechina.csdn.net/EricLee/e3d_handpose_x "e3d_handpose_x")


因为原作用用到的mono里面有些脚本文件是基于python2的，我的环境是Python3的，有些包已经变了，我只是稍作修改，在我的电脑上能运行起来。并做个备份
若要运行下来，需要修改mano/webuser/smpl_handpca_wrapper_HAND_only.py这个文件的
第59行
    `fname_or_dict = 'D:/AI-LAB/e3d_handpose_x-master/mano/models/MANO_RIGHT.pkl'`

和第96行
来指定自己模型的目录
# Easy 3D HandPose X  
Easy 3D HandPose，pytorch，单目相机的手三维姿态估计

## 项目介绍   
* 注意：该项目前向推理用到项目包括（里面全是原作者的一些项目）：  
- [x] 手检测：https://codechina.csdn.net/EricLee/yolo_v3
- [x] 手二维关键点检测：https://codechina.csdn.net/EricLee/handpose_x
- [x] Manopth：https://github.com/hassony2/manopth
- [x] Easy 3D HandPose X：https://codechina.csdn.net/EricLee/e3d_handpose_x

* 视频示例：  
 ![video](https://github.com/BiggerBinBin/e3d_handpose_x-master/blob/master/samples/sample.gif)    
* [完整示例视频](https://www.bilibili.com/video/BV1j64y1r7da/)       
## 项目推理 PipeLine
 ![pipeline](https://github.com/BiggerBinBin/e3d_handpose_x-master/blob/master/samples/e3d.jpg)  
## 项目配置  
### 1、软件  
* 作者开发环境：  
* Python 3.8  
* PyTorch >= 1.5.1  
* opencv-python  
* open3d  
### 2、硬件  
* 笔记本自带摄像头

## 项目使用方法  
### [1]准备左右手3D建模资源（MANO Hand Model）和原作者的一些资源  
链接：[https://pan.baidu.com/s/1TE5ig27jjvF690ZezhgO5A ](https://pan.baidu.com/s/1TE5ig27jjvF690ZezhgO5A )
提取码：8md6 
### [2]模型训练  
#### 根目录下运行命令： python train.py       (注意脚本内相关参数配置 )   

### [3]模型推理  
#### 根目录下运行命令：
#### 1) 图片推理脚本 ：python inference.py        (注意脚本内相关参数配置 )

#### 2) 相机推理脚本 ： python yolo_inference.py        (注意脚本内相关参数配置 )  
####    注意：目前推理为 “预发版本”，只支持画面中出现一只手会进行三维姿态估计，并只支持右手姿态估计。   

#####    第 1 步：确定电脑连接相机。
#####	第2步：下载mono和if_package两个文件夹到目录,大概文件夹的结构如下：
```
e3d_handpose_x/
├─components
│  ├─hand_detect
│  │  ├─utils
│  │  │  └─__pycache__
│  │  └─__pycache__
│  └─hand_keypoints
│      ├─models
│      │  └─__pycache__
│      ├─utils
│      │  └─__pycache__
│      └─__pycache__
├─e3d_data_iter
│  └─__pycache__
├─if_package
├─loss
├─mano
│  ├─models
│  ├─webuser
│  │  ├─hello_world
│  │  └─__pycache__
│  └─__pycache__
├─manopth
│  └─__pycache__
├─models
│  └─__pycache__
├─samples
├─utils
│  └─__pycache__
└─__pycache__
```
#####   第 3 步：运行脚本：python yolo_inference.py
####   注意：运行出错，注意看log报的错误，尽量自行解决，思考尝试解决不了，issue提问。

## 联系方式 （Contact）  
* E-mails: 1091313282@qq.com   

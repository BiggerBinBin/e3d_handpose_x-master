#-*-coding:utf-8-*-
# date:2021-06-15
# Author: Eric.Lee
# function: handpose 3D Yolo_v3 Detect Inference

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F

from models.resnet import resnet18,resnet34,resnet50,resnet101
from e3d_data_iter.datasets import letterbox,get_heatmap
import sys
sys.path.append("./components/") # 添加模型组件路径
from hand_keypoints.handpose_x import handpose_x_model,draw_bd_handpose_c
from hand_detect.yolo_v3_hand import yolo_v3_hand_model

from utils.common_utils import *
import copy

from utils import func, bone, AIK, smoother
from utils.LM_new import LM_Solver
from op_pso import PSO
import open3d
from mpl_toolkits.mplot3d import Axes3D
from manopth import manolayer
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Hand Pose 3D Inference')
    parser.add_argument('--model_path', type=str, default = './if_package/e3d_handposex-resnet_50-size-128-loss-wing_loss-20210620.pth',
        help = 'model_path') # e3d handpose 模型路径
    parser.add_argument('--detect_model_path', type=str, default = './if_package/hand_detect_416-20210606.pt',
        help = 'model_path') # detect 模型路径
    parser.add_argument('--handpose_x2d_model_path', type=str, default = './if_package/handposex_2d_resnet_50-size-256-wingloss102-0.119.pth',
        help = 'model_path') # 手2维关键点 模型路径
    parser.add_argument('--model', type=str, default = 'resnet_50',
        help = '''model : resnet_18,resnet_34,resnet_50,resnet_101''') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 63,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './image/',
        help = 'test_path') # 测试图片路径
    parser.add_argument('--img_size', type=tuple , default = (128,128),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path # 测试图片文件夹路径
    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_18':
        model_ = resnet18(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_ = resnet101(num_classes = ops.num_classes,img_size=ops.img_size[0])

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式
    # print(model_)# 打印模型结构
    # 加载测试模型
    if os.access(ops.model_path,os.F_OK):# checkpoint
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

    #----------------- 构建 handpose_x 2D关键点检测模型
    handpose_2d_model = handpose_x_model(model_path = ops.handpose_x2d_model_path)
    #----------------- 构建 yolo 检测模型
    hand_detect_model = yolo_v3_hand_model(model_path = ops.detect_model_path,model_arch = "yolo",conf_thres = 0.3)
    # hand_detect_model = yolo_v3_hand_model()

    #----------------- 构建 manopth
    g_side = "right"
    print('load model finished')
    pose, shape = func.initiate("zero")
    pre_useful_bone_len = np.zeros((1, 15)) # 骨架点信息
    solver = LM_Solver(num_Iter=99, th_beta=shape.cpu(), th_pose=pose.cpu(), lb_target=pre_useful_bone_len,
                       weight=1e-5)
    pose0 = torch.eye(3).repeat(1, 16, 1, 1)

    mano = manolayer.ManoLayer(flat_hand_mean=True,
                               side=g_side,
                               mano_root='./mano/models',
                               use_pca=False,
                               root_rot_mode='rotmat',
                               joint_rot_mode='rotmat')
    print('start ~')
    point_fliter = smoother.OneEuroFilter(23.0, 0.0)
    mesh_fliter = smoother.OneEuroFilter(23.0, 0.0)
    shape_fliter = smoother.OneEuroFilter(1.5, 0.0)
    #--------------------------- 配置点云
    view_mat = np.array([[1.0, 0.0, 0.0],
                         [0.0, -1.0, 0],
                         [0.0, 0, -1.0]])
    mesh = open3d.geometry.TriangleMesh()
    hand_verts, j3d_recon = mano(pose0, shape.float())
    mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
    hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
    mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    viewer = open3d.visualization.Visualizer()
    viewer.create_window(width=800, height=800, window_name='HandPose3d_Mesh')
    viewer.add_geometry(mesh)
    viewer.update_renderer()
    renderOptions = viewer.get_render_option ()
    renderOptions.background_color = np.asarray([120/255,120/255,120/255]) # 设置背景颜色
    # axis_pcd = open3d.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # vis.add_geometry(axis_pcd)
    pts_flag = False
    if pts_flag:
        test_pcd = open3d.geometry.PointCloud()  # 定义点云
        viewer.add_geometry(test_pcd)

    print('start pose estimate')

    pre_uv = None
    shape_time = 0
    opt_shape = None
    shape_flag = True
    #---------------------------------------------------------------- 预测图片

    with torch.no_grad():
        idx = 0
        cap = cv2.VideoCapture(0) #一般usb默认相机号为 0，如果没有相机无法启动，如果相机不为0需要自行确定其编号。

        while True:
            ret, img_o = cap.read()# 获取相机图像
            if ret == True:# 如果 ret 返回值为 True，显示图片
                img_yolo_x = img_o.copy()
                hand_bbox =hand_detect_model.predict(img_yolo_x,vis = False) # 检测手，获取手的边界框
                if len(hand_bbox) == 1:
                    #----------------------------------
                    finger_index = None # 食指UI的二维坐标
                    finger_thumb = None # 大拇UI指的二维坐标
                    #----------------------------------
                    x_min,y_min,x_max,y_max,_ = hand_bbox[0]

                    w_ = max(abs(x_max-x_min),abs(y_max-y_min))
                    w_ = w_*1.6
                    x_mid = (x_max+x_min)/2
                    y_mid = (y_max+y_min)/2
                    #
                    x1,y1,x2,y2 = int(x_mid-w_/2),int(y_mid-w_/2),int(x_mid+w_/2),int(y_mid+w_/2)
                    #
                    x1 = int(np.clip(x1,0,img_o.shape[1]-1))
                    y1 = int(np.clip(y1,0,img_o.shape[0]-1))
                    x2 = int(np.clip(x2,0,img_o.shape[1]-1))
                    y2 = int(np.clip(y2,0,img_o.shape[0]-1))

                    img = img_o[y1:y2,x1:x2]
                else:
                    continue
                #--------------------------------
                img_show = img.copy() # 用于显示使用
                pts_2d_ = handpose_2d_model.predict(img.copy()) # handpose_2d predict
                pts_2d_hand = {}

                kps_min_x,kps_min_y,kps_max_x,kps_max_y = 65535.,65535.,0.,0.
                for ptk in range(int(pts_2d_.shape[0]/2)):

                    xh = pts_2d_[ptk*2+0]*float(img.shape[1])
                    yh = pts_2d_[ptk*2+1]*float(img.shape[0])
                    pts_2d_hand[str(ptk)] = {
                        "x":xh,
                        "y":yh,
                        }
                    kps_min_x = (xh+x1) if (xh+x1)<kps_min_x else kps_min_x
                    kps_min_y = (yh+y1) if (yh+y1)<kps_min_y else kps_min_y
                    kps_max_x = (xh+x1) if (xh+x1)>kps_max_x else kps_max_x
                    kps_max_y = (yh+y1) if (yh+y1)>kps_max_y else kps_max_y
                    if ptk == 3 or ptk == 4:
                        if finger_thumb is None:
                            finger_thumb = (int(xh+x1),int(yh+y1))
                        else:
                            finger_thumb = (int((xh+x1+finger_thumb[0])/2),int((yh+y1+finger_thumb[1])/2))
                    if ptk == 7 or ptk == 8:
                        if finger_index is None:
                            finger_index = (int(xh+x1),int(yh+y1))
                        else:
                            finger_index = (int((xh+x1+finger_index[0])/2),int((yh+y1+finger_index[1])/2))
                        # cv2.circle(img_show, finger_index, 9, (25,160,255),-1)
                    if ops.vis:
                        cv2.circle(img_show, (int(xh),int(yh)), 4, (255,50,60),-1)
                        cv2.circle(img_show, (int(xh),int(yh)), 3, (25,160,255),-1)
                hand2d_kps_bbox = (int(kps_min_x),int(kps_min_y),int(kps_max_x),int(kps_max_y)) # 手关键点边界框
                if ops.vis:
                    draw_bd_handpose_c(img_show,pts_2d_hand,0,0,2)
                    cv2.namedWindow("handpose_2d",0)
                    cv2.imshow("handpose_2d",img_show)

                #--------------------------------
                img_lbox,ratio, dw, dh = letterbox(img.copy(), height=ops.img_size[0], color=(0,0,0))
                # if ops.vis:
                #     cv2.namedWindow("letterbox",0)
                #     cv2.imshow("letterbox",img_lbox)

                #-------------------------------- get heatmap
                x1y1x2y2 = 0,0,0,0
                offset_x1,offset_y1 = 0,0
                hm,hm_w = get_heatmap(img_lbox.copy(),x1y1x2y2,pts_2d_hand,ratio, dw, dh,offset_x1,offset_y1,vis=False)
                if ops.vis:
                    cv2.namedWindow("hm_w",0)
                    cv2.imshow("hm_w",hm_w)

                #--------------------------------
                img_fix_size = img_lbox.astype(np.float32)

                img_fix_size_r = img_fix_size.astype(np.float32)
                img_fix_size_r = (img_fix_size_r-128.)/256.
                #--------------------------------------------------
                image_fusion = np.concatenate((img_fix_size_r,hm),axis=2)
                image_fusion = image_fusion.transpose(2, 0, 1)
                image_fusion = torch.from_numpy(image_fusion)
                image_fusion = image_fusion.unsqueeze_(0)
                if use_cuda:
                    image_fusion = image_fusion.cuda()  # (bs, channel, h, w)
                # print("image_fusion size : {}".format(image_fusion.size()))

                #--------------------------------  # handpose_3d predict
                pre_ = model_(image_fusion.float()) # 模型推理
                output = pre_.cpu().detach().numpy()
                output = np.squeeze(output)
                # print("handpose_3d output shape : {}".format(output.shape))

                pre_3d_joints = output.reshape((21,3))
                # print("pre_3d_joints shape : {}".format(pre_3d_joints.shape))

                if g_side == "left":
                    print("------------------->>. left")
                    pre_3d_joints[:,0] *=(-1.)
                pre_3d_joints = torch.tensor(pre_3d_joints).squeeze(0)
                pre_3d_joints= pre_3d_joints.cuda()
                # print(pre_3d_joints.size())
                #--------------------------------------------------------------------
                # now_uv = result['uv'].clone().detach().cpu().numpy()[0, 0]
                # now_uv = now_uv.astype(np.float)
                trans = np.zeros((1, 3))
                # trans[0, 0:2] = now_uv - 16.0
                trans = trans / 16.0
                new_tran = np.array([[trans[0, 1], trans[0, 0], trans[0, 2]]])
                pre_joints = pre_3d_joints.clone().detach().cpu().numpy()

                flited_joints = point_fliter.process(pre_joints)

                # fliter_ax.cla()
                #
                # filted_ax = vis.plot3d(flited_joints + new_tran, fliter_ax)
                # pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")

                pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")

                NGEN = 0 # PSO 迭代次数
                popsize = 100
                low = np.zeros((1, 10)) - 3.0
                up = np.zeros((1, 10)) - 2.0
                parameters = [NGEN, popsize, low, up]
                pso = PSO(parameters, pre_useful_bone_len.reshape((1, 15)),g_side)
                pso.main(solver)
                if True:#opt_shape is None:
                    opt_shape = pso.ng_best
                    opt_shape = shape_fliter.process(opt_shape)

                opt_tensor_shape = torch.tensor(opt_shape, dtype=torch.float)
                _, j3d_p0_ops = mano(pose0, opt_tensor_shape)
                template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0  # template, m 21*3
                ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
                j3d_pre_process = pre_joints * ratio  # template, m
                j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
                pose_R = AIK.adaptive_IK(template, j3d_pre_process)
                pose_R = torch.from_numpy(pose_R).float()
                #  reconstruction
                hand_verts, j3d_recon = mano(pose_R, opt_tensor_shape.float())
                hand_verts[:,:,:] = hand_verts[:,:,:]*(0.503)
                # print(j3d_recon.size())

                mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
                hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
                hand_verts = mesh_fliter.process(hand_verts)
                hand_verts = np.matmul(view_mat, hand_verts.T).T
                if g_side == "right":
                    hand_verts[:, 0] = hand_verts[:, 0] - 40
                else:
                    hand_verts[:, 0] = hand_verts[:, 0] + 40
                hand_verts[:, 1] = hand_verts[:, 1] - 0
                mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
                hand_verts = hand_verts - 100 * mesh_tran

                mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
                # mesh.paint_uniform_color([252 / 255, 224 / 255, 203 / 255])
                # mesh.paint_uniform_color([238 / 255, 188 / 255, 158 / 255])
                mesh.paint_uniform_color([87 / 255, 131 / 255, 235 / 255])
                mesh.compute_triangle_normals()
                mesh.compute_vertex_normals()
                #-----------
                if pts_flag:
                    if False:
                        j3d_ = j3d_recon.detach().cpu().numpy()
                        j3d_[0][:,1] *=(-1.)
                        # j3d_[0][:,0] +=trans[0,0]
                        j3d_[0] = j3d_[0] - 100 * mesh_tran
                        j3d_[0][:,0] -=50
                        j3d_[0][:,1] -=30
                        # print(j3d_.shape,j3d_)
                        test_pcd.points = open3d.utility.Vector3dVector(j3d_[0])  # 定义点云坐标位置
                    else:
                        # 778*3
                        print("hand_verts shape : {}".format(hand_verts.shape))
                        # a = np.concatenate((
                        #     hand_verts[38:40,:],
                        #     hand_verts[40:100,:]
                        #     ),axis=0)
                        test_pcd.points = open3d.utility.Vector3dVector(hand_verts)
                        pre_joints[:,1] *=-1.
                        pre_joints = pre_joints*70
                        pre_joints[:,1] -= 40
                        pre_joints[:,0] -= 0
                        # print("pre_joints",pre_joints.shape)
                        # test_pcd.points = open3d.utility.Vector3dVector(pre_joints)
                        # test_pcd.points = open3d.utility.Vector3dVector(pre_joints[1,:].reshape(1,3))
                        # rgb = np.asarray([250,0,250])
                        # rgb_t = np.transpose(rgb)
                        # test_pcd.colors = open3d.utility.Vector3dVector(rgb_t.astype(np.float) / 255.0)
                # print("hand_verts shape",hand_verts)
                #-----------
                viewer.update_geometry(mesh)
                if pts_flag:
                    viewer.update_geometry(test_pcd)
                viewer.poll_events()
                viewer.update_renderer()
                #---------------------------------------------------------------
                image_open3d = viewer.capture_screen_float_buffer(False)
                # viewer.capture_screen_image("open3d.jpg", False)
                # depth = vis.capture_depth_float_buffer(False)
                image_3d = viewer.capture_screen_float_buffer(False)
                image_3d = np.asarray(image_3d)
                image_3d = image_3d*255
                image_3d = np.clip(image_3d,0,255)
                image_3d = image_3d.astype(np.uint8)
                image_3d = cv2.cvtColor(image_3d, cv2.COLOR_RGB2BGR)

                # print(image_3d.shape)
                mask_0 = np.where(image_3d[:,:,0]!=120,1,0)
                mask_1 = np.where(image_3d[:,:,1]!=120,1,0)
                mask_2 = np.where(image_3d[:,:,2]!=120,1,0)
                img_mask = np.logical_or(mask_0,mask_1)
                img_mask = np.logical_or(img_mask,mask_2)
                # cv2.namedWindow("img_mask",0)
                # cv2.imshow("img_mask",img_mask.astype(np.float))

                locs = np.where(img_mask != 0)
                xx1 = np.min(locs[1])
                xx2 = np.max(locs[1])
                yy1 = np.min(locs[0])
                yy2 = np.max(locs[0])
                # cv2.rectangle(image_3d, (xx1,yy1), (xx2,yy2), (255,0,255), 5) # 绘制image_3d
                model_hand_w = (xx2-xx1)
                model_hand_h = (yy2-yy1)
                #----------
                cv2.namedWindow("image_3d",0)
                cv2.imshow("image_3d",image_3d)


                # cv2.rectangle(img_yolo_x, (hand2d_kps_bbox[0],hand2d_kps_bbox[1]),
                #     (hand2d_kps_bbox[2],hand2d_kps_bbox[3]), (0,165,255), 3) # 绘制2d 手关键点
                # scale_ = ((x_max-x_min)/(xx2-xx1) + (y_max-y_min)/(yy2-yy1))/2.*1.01
                scale_ = ((hand2d_kps_bbox[2]-hand2d_kps_bbox[0])/(xx2-xx1)
                    + (hand2d_kps_bbox[3]-hand2d_kps_bbox[1])/(yy2-yy1))/2.*1.2
                w_3d_ = (xx2-xx1)*scale_
                h_3d_ = (yy2-yy1)*scale_
                x_mid_3d = (xx1+xx2)/2.
                y_mid_3d = (yy1+yy2)/2.

                x_mid,y_mid = int(x_mid),int(y_mid)
                x1,y1,x2,y2 = int(x_mid-w_3d_/2.),int(y_mid-h_3d_/2.),int(x_mid+w_3d_/2.),int(y_mid+h_3d_/2.)
                crop_ = image_3d[yy1:yy2,xx1:xx2]
                crop_mask = (img_mask[yy1:yy2,xx1:xx2].astype(np.float)*255).astype(np.uint8)
                w_r,h_r = int(crop_.shape[1]*scale_/2),int(crop_.shape[0]*scale_/2)
                crop_ = cv2.resize(crop_, (w_r*2, h_r*2))
                crop_mask = cv2.resize(crop_mask, (w_r*2, h_r*2))
                crop_mask = np.where(crop_mask[:,:]>0.,1.,0.)
                crop_mask = np.expand_dims(crop_mask, 2)

                try:
                    #------------
                    img_ff = img_yolo_x[int(y_mid - h_r ):int(y_mid + h_r ),int(x_mid - w_r ):int(x_mid + w_r ),:]*(1.-crop_mask) + crop_*crop_mask
                    img_yolo_x[int(y_mid - h_r ):int(y_mid + h_r ),int(x_mid - w_r ):int(x_mid + w_r ),:] = img_ff.astype(np.uint8)
                except:
                    continue

                real_hand_w = w_r*2
                real_hand_h = h_r*2
                depth_z = (model_hand_h/real_hand_h  + model_hand_w/real_hand_w)/2.# 相对深度 z
                #
                cv2.putText(img_yolo_x, " Relative Depth_Z :{:.3f} ".format(depth_z), (4,42),cv2.FONT_HERSHEY_DUPLEX, 1.1, (55, 0, 220),7)
                cv2.putText(img_yolo_x, " Relative Depth_Z :{:.3f} ".format(depth_z), (4,42),cv2.FONT_HERSHEY_DUPLEX, 1.1, (25, 180, 250),1)


                cv2.namedWindow("img_yolo_x",0)
                cv2.imshow("img_yolo_x",img_yolo_x)

                # x_mid = (x_max+x_min)/2
                # y_mid = (y_max+y_min)/2

                if cv2.waitKey(1) == 27:
                    break
            else:
                break

    cv2.destroyAllWindows()

    print('well done ')

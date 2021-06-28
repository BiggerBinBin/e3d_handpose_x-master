#-*-coding:utf-8-*-
# date:2021-06-15
# Author: Eric.Lee
# function: handpose 3D Inference

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
    parser.add_argument('--model_path', type=str, default = './model_exp/2021-06-16_02-09-37/resnet_50-size-256-loss-wing_loss-model_epoch-732.pth',
        help = 'model_path') # 模型路径
    parser.add_argument('--model', type=str, default = 'resnet_50',
        help = '''model : resnet_18,resnet_34,resnet_50,resnet_101''') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 63,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './image/',
        help = 'test_path') # 测试图片路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
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
    handpose_2d_model = handpose_x_model()

    #----------------- 构建 manopth
    g_side = "right"
    print('load model finished')
    pose, shape = func.initiate("zero")
    pre_useful_bone_len = np.zeros((1, 15)) # 骨架点信息
    solver = LM_Solver(num_Iter=666, th_beta=shape.cpu(), th_pose=pose.cpu(), lb_target=pre_useful_bone_len,
                       weight=1e-5)
    pose0 = torch.eye(3).repeat(1, 16, 1, 1)

    mano = manolayer.ManoLayer(flat_hand_mean=True,
                               side=g_side,
                               mano_root='./mano/models',
                               use_pca=False,
                               root_rot_mode='rotmat',
                               joint_rot_mode='rotmat')
    print('start ~')
    point_fliter = smoother.OneEuroFilter(4.0, 0.0)
    mesh_fliter = smoother.OneEuroFilter(4.0, 0.0)
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
    viewer.create_window(width=640, height=640, window_name='HandPose3d_Mesh')
    viewer.add_geometry(mesh)
    viewer.update_renderer()
    renderOptions = viewer.get_render_option ()
    renderOptions.background_color = np.asarray([120/255,120/255,120/255]) # 设置背景颜色
    # axis_pcd = open3d.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # vis.add_geometry(axis_pcd)
    pts_flag = True
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
        for file in os.listdir(ops.test_path):
            if '.jpg' not in file:
                continue
            idx += 1
            print('{}) image : {}'.format(idx,file))
            img = cv2.imread(ops.test_path + file)
            #--------------------------------
            img_show = img.copy() # 用于显示使用
            pts_2d_ = handpose_2d_model.predict(img.copy()) # handpose_2d predict
            pts_2d_hand = {}
            for ptk in range(int(pts_2d_.shape[0]/2)):

                xh = pts_2d_[ptk*2+0]*float(img.shape[1])
                yh = pts_2d_[ptk*2+1]*float(img.shape[0])
                pts_2d_hand[str(ptk)] = {
                    "x":xh,
                    "y":yh,
                    }
                if ops.vis:
                    cv2.circle(img_show, (int(xh),int(yh)), 4, (255,50,60),-1)
                    cv2.circle(img_show, (int(xh),int(yh)), 3, (25,160,255),-1)
            if ops.vis:
                draw_bd_handpose_c(img_show,pts_2d_hand,0,0,2)
                cv2.namedWindow("handpose_2d",0)
                cv2.imshow("handpose_2d",img_show)

            #--------------------------------
            img_lbox,ratio, dw, dh = letterbox(img.copy(), height=ops.img_size[0], color=(0,0,0))
            if ops.vis:
                cv2.namedWindow("letterbox",0)
                cv2.imshow("letterbox",img_lbox)

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
            print("image_fusion size : {}".format(image_fusion.size()))


            #--------------------------------  # handpose_3d predict
            pre_ = model_(image_fusion.float()) # 模型推理
            output = pre_.cpu().detach().numpy()
            output = np.squeeze(output)
            print("handpose_3d output shape : {}".format(output.shape))

            pre_3d_joints = output.reshape((21,3))
            print("pre_3d_joints shape : {}".format(pre_3d_joints.shape))

            if g_side == "left":
                print("------------------->>. left")
                pre_3d_joints[:,0] *=(-1.)
            pre_3d_joints = torch.tensor(pre_3d_joints).squeeze(0)
            pre_3d_joints= pre_3d_joints.cuda()
            print(pre_3d_joints.size())
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
            hand_verts[:,:,:] = hand_verts[:,:,:]*(0.85)
            # print(j3d_recon.size())

            mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
            hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
            hand_verts = mesh_fliter.process(hand_verts)
            hand_verts = np.matmul(view_mat, hand_verts.T).T
            if g_side == "right":
                hand_verts[:, 0] = hand_verts[:, 0] - 80
            else:
                hand_verts[:, 0] = hand_verts[:, 0] + 80
            hand_verts[:, 1] = hand_verts[:, 1] - 0
            mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
            hand_verts = hand_verts - 100 * mesh_tran

            mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
            # mesh.paint_uniform_color([252 / 255, 224 / 255, 203 / 255])
            mesh.paint_uniform_color([238 / 255, 188 / 255, 158 / 255])
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
                    # test_pcd.points = open3d.utility.Vector3dVector(hand_verts)
                    pre_joints[:,1] *=-1.
                    pre_joints = pre_joints*70
                    pre_joints[:,1] -= 40
                    pre_joints[:,0] -= 0
                    print("pre_joints",pre_joints.shape)
                    test_pcd.points = open3d.utility.Vector3dVector(pre_joints)
                    # test_pcd.points = open3d.utility.Vector3dVector(pre_joints[1,:].reshape(1,3))
                    # rgb = np.asarray([250,0,250])
                    # rgb_t = np.transpose(rgb)
                    # test_pcd.colors = open3d.utility.Vector3dVector(rgb_t.astype(np.float) / 255.0)
            # print("hand_verts shape",hand_verts)
            # x_min,y_min,x_max,y_max = 65535,65535,0,0
            # for i in range(hand_verts.shape[0]):
            #     x_,y_,z_ = hand_verts[i]
            #     x_min = x_ if x_min>x_ else x_min
            #     y_min = y_ if y_min>y_ else y_min
            #     x_max = x_ if x_max<x_ else x_max
            #     y_max = y_ if y_max<y_ else y_max
            # print("x_min,y_min,x_max,y_max : ",x_min,y_min,x_max,y_max)
            #-----------
            viewer.update_geometry(mesh)
            if pts_flag:
                viewer.update_geometry(test_pcd)
            viewer.poll_events()
            viewer.update_renderer()
            #----------
            cv2.namedWindow("img",0)
            cv2.imshow("img",img)

            if cv2.waitKey(1) == 27:
                break

    cv2.destroyAllWindows()

    print('well done ')

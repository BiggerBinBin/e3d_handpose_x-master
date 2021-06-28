#-*-coding:utf-8-*-
# date:2021-06-15
# Author: Eric.Lee
# function: easy 3d handpose data iter
import glob
import math
import os
import random

from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
#----------------------
import torch
from manopth import manolayer
# from model.detnet import detnet
from utils import func, bone, AIK, smoother
from utils.LM_new import LM_Solver
import numpy as np
import matplotlib.pyplot as plt
from utils import vis
from op_pso import PSO
import open3d
from mpl_toolkits.mplot3d import Axes3D
import time
#----------------------

def draw_handpose_2d(img_,hand_,x,y,thick = 3):
    # thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)

def img_agu_channel_same(img_):
    img_a = np.zeros(img_.shape, dtype = np.uint8)
    gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)
    img_a[:,:,0] =gray
    img_a[:,:,1] =gray
    img_a[:,:,2] =gray

    return img_a
# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst

def letterbox(img, height=416, augment=False, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resize img
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
      np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap
def get_heatmap(img_fix_size,x1y1x2y2,handpose_2d,ratio, dw, dh,offset_x1=0,offset_y1=0,radius=20,vis = False):
    num_objs = 21

    hm = np.zeros((img_fix_size.shape[0],img_fix_size.shape[1],num_objs), dtype=np.float32)
    draw_gaussian = draw_msra_gaussian if False else draw_umich_gaussian

    for k in range(num_objs):
        x,y = (handpose_2d[str(k)]["x"]-offset_x1)*ratio+round(dw - 0.1),(handpose_2d[str(k)]["y"]-offset_y1)*ratio+round(dh - 0.1)

        draw_gaussian(hm[:,:,k], (x,y), radius)

        if vis:
            # print("x,y : ",x,y)
            cv2.namedWindow("hm",0)
            cv2.imshow("hm",hm[:,:,k])
            cv2.circle(img_fix_size, (int(x),int(y)), 3, (250,60,255),-1)
            cv2.namedWindow("fix_size",0)
            cv2.imshow("fix_size",img_fix_size)
            cv2.waitKey(1)
            # print("------------------------>>>")
    hm_w = hm.max(axis=2)
    if vis:
        cv2.namedWindow("hm_w",0)
        cv2.imshow("hm_w",hm_w)
        # cv2.waitKey(1)
        # print(hm_w.size)
    return hm,hm_w
class LoadImagesAndLabels(Dataset):
    def __init__(self, ops, img_size=(256,256), flag_agu = False,vis = False):
        print('img_size (height,width) : ',img_size[0],img_size[1])
        print("train_path : {}".format(ops.train_path))
        g_side = "right"
        path = ops.train_path

        #-----------------------
        if vis:
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
        #-----------------------------------------------------------------------
        file_list = []
        label_list = []
        bbox_list = []
        handpose_2d_x1y1x2y2_list = []
        handpose_2d_pts_hand_list = []
        handpose_3d_xyz_list = []
        idx = 0
        for f_ in os.listdir(path):
            if ".jpg" in f_:
                thr = 0
                num_ = int(f_.split("_")[-1].replace(".jpg",""))
                file_img = path + f_
                file_json = file_img.replace("_{}.jpg".format(num_),"_{}.json".format(num_+thr))
                if not os.access(file_json,os.F_OK):
                    continue
                #-----------------------------
                file_list.append(file_img)
                label_list.append(file_json)
                #-----------------------------
                # print(file_json)
                f = open(file_json, encoding='utf-8')#读取 json文件
                dict_x = json.load(f)
                f.close()
                # print(dict_x)
                #--------------------
                if vis:
                    img = cv2.imread(file_img)
                if g_side == "left":
                    img = cv2.flip(img,1)
                bbox = dict_x["bbox"]
                handpose_2d = dict_x["handpose_2d"]
                #-----------------
                x1_,y1_,x2_,y2_ = handpose_2d["x1y1x2y2"]
                x1_,y1_,x2_,y2_ = int(x1_),int(y1_),int(x2_),int(y2_)
                gt_3d_joints = dict_x["handpose_3d_xyz"]
                #
                handpose_2d_x1y1x2y2_list.append((x1_,y1_,x2_,y2_))
                handpose_2d_pts_hand_list.append(handpose_2d["pts_hand"])
                handpose_3d_xyz_list.append(gt_3d_joints)
                if vis:
                    img_fix_size,ratio, dw, dh = letterbox(img[y1_:y2_,x1_:x2_], height=img_size[0], color=(0,0,0))

                    hm,hm_w = get_heatmap(img_fix_size,handpose_2d["x1y1x2y2"],handpose_2d["pts_hand"],ratio, dw, dh,vis=False)

                    cv2.namedWindow("fix_size",0)
                    cv2.imshow("fix_size",img_fix_size)

                    hm_w = np.expand_dims(hm_w,2)

                    print("hm.shape : {}".format(hm.shape))
                    print("hm_w.shape : {}".format(hm_w.shape))
                    print("img_fix_size.shape : {}".format(img_fix_size.shape))
                    img_fix_size_r = img_fix_size.astype(np.float32)
                    img_fix_size_r = (img_fix_size_r-128.)/256.
                    #--------------------------------------------------
                    image_fusion = np.concatenate((img_fix_size_r,hm),axis=2)
                    print(" A image_fusion.shape : {}".format(image_fusion.shape))

                    image_fusion = image_fusion.transpose(2, 0, 1)
                    print(" B image_fusion.shape : {}".format(image_fusion.shape))

                    image_fusion = np.expand_dims(image_fusion,0)
                    print(" C image_fusion.shape : {}".format(image_fusion.shape))

                    # img_fix_size_r = np.expand_dims(img_fix_size_r,0)
                    # print(hm.shape ," 《《-------------》》",img_fix_size_r.shape)
                    # #--------------------------------------------------
                    # image_fusion = np.concatenate((img_fix_size_r,hm_w),axis=0)


                    #-----------------
                    cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0,255), 5) # 绘制空心矩形

                    pts_hand2d = handpose_2d["pts_hand"]
                    draw_handpose_2d(img,pts_hand2d,x1_,y1_,2)

                    #---------------------------------
                    gt_3d_joints= np.array(gt_3d_joints)
                    print(gt_3d_joints.shape)
                    if g_side == "left":
                        print("------------------->>. left")
                        gt_3d_joints[:,0] *=(-1.)
                    gt_3d_joints = torch.tensor(gt_3d_joints).squeeze(0)
                    gt_3d_joints= gt_3d_joints.cuda()
                    print(gt_3d_joints.size())
                    #------------------------------
                    # now_uv = result['uv'].clone().detach().cpu().numpy()[0, 0]
                    # now_uv = now_uv.astype(np.float)
                    trans = np.zeros((1, 3))
                    # trans[0, 0:2] = now_uv - 16.0
                    trans = trans / 16.0
                    new_tran = np.array([[trans[0, 1], trans[0, 0], trans[0, 2]]])
                    gt_3d_joints = gt_3d_joints.clone().detach().cpu().numpy()

                    flited_joints = point_fliter.process(gt_3d_joints)

                    # fliter_ax.cla()
                    #
                    # filted_ax = vis.plot3d(flited_joints + new_tran, fliter_ax)
                    pre_useful_bone_len = bone.caculate_length(gt_3d_joints, label="useful")

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
                    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(gt_3d_joints[9] - gt_3d_joints[0])
                    j3d_pre_process = gt_3d_joints * ratio  # template, m
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
                            gt_3d_joints[:,1] *=-1.
                            gt_3d_joints = gt_3d_joints*70
                            gt_3d_joints[:,1] -= 40
                            gt_3d_joints[:,0] -= 0
                            print("gt_3d_joints",gt_3d_joints.shape)
                            test_pcd.points = open3d.utility.Vector3dVector(gt_3d_joints)
                            # test_pcd.points = open3d.utility.Vector3dVector(gt_3d_joints[1,:].reshape(1,3))
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

                    cv2.namedWindow("img",0)
                    cv2.imshow("img",img)
                    cv2.waitKey(1)

        #-----------------------------------------------------------------------
        if vis:
            cv2.destroyAllWindows()

        #
        print()
        self.files = file_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.vis = vis

        # label_list = []
        self.bbox_list = bbox_list
        self.x1y1x2y2_2d_list = handpose_2d_x1y1x2y2_list
        self.pts_hand_2d_list = handpose_2d_pts_hand_list
        self.xyz_3d_list = handpose_3d_xyz_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # gt_3d_joints = dict_x["handpose_3d_xyz"]
        # #
        # handpose_2d_x1y1x2y2_list.append((x1_,y1_,x2_,y2_))
        # handpose_2d_pts_hand_list.append(handpose_2d["pts_hand"])
        # handpose_3d_xyz_list.append(gt_3d_joints)
        img_path = self.files[index]
        x1y1x2y2 = self.x1y1x2y2_2d_list[index]
        pts_hand = self.pts_hand_2d_list[index]
        gt_3d_joints = self.xyz_3d_list[index]
        img = cv2.imread(img_path)  # BGR

        x1_,y1_,x2_,y2_ = x1y1x2y2



        hand_w = int((x2_-x1_)/2)
        hand_h = int((y2_-y1_)/2)
        offset_x1 = random.randint(-hand_w,int(hand_w/6))
        offset_y1 = random.randint(-hand_h,int(hand_h/6))
        offset_x2 = random.randint(-int(hand_w/6),hand_w)
        offset_y2 = random.randint(-int(hand_h/6),hand_h)
        # print(" A : x1_,y1_,x2_,y2_ ： ",x1_,y1_,x2_,y2_)
        x1_new = x1_+offset_x1
        y1_new = y1_+offset_y1
        x2_new = x2_+offset_x2
        y2_new = y2_+offset_y2

        x1_new = np.clip(x1_,0,img.shape[1]-1)
        x2_new = np.clip(x2_,0,img.shape[1]-1)
        y1_new = np.clip(y1_,0,img.shape[0]-1)
        y2_new = np.clip(y2_,0,img.shape[0]-1)

        offset_x1 = x1_new - x1_
        offset_y1 = y1_new - y1_
        offset_x2 = x2_new - x2_
        offset_y2 = y2_new - y2_
        # print(" B : x1_,y1_,x2_,y2_ ： ",x1_,y1_,x2_,y2_)
        x1_ = x1_new
        y1_ = y1_new
        x2_ = x2_new
        y2_ = y2_new
        #-------------------------------------
        # if self.vis:
        #     aa = img[y1_:y2_,x1_:x2_]
        #     for k in range(21):
        #         x,y = (pts_hand[str(k)]["x"]-offset_x1),(pts_hand[str(k)]["y"]-offset_y1)
        #
        #
        #         cv2.circle(aa, (int(x),int(y)), 3, (250,60,255),-1)
        #     cv2.namedWindow("fix_size_a",0)
        #     cv2.imshow("fix_size_a",aa)
        #-------------------------------------
        # print("self.img_size : ",self.img_size)
        img_,ratio, dw, dh = letterbox(img[y1_:y2_,x1_:x2_], height=self.img_size[0], color=(0,0,0))

        hm,hm_w = get_heatmap(img_,x1y1x2y2,pts_hand,ratio, dw, dh,offset_x1,offset_y1,vis=self.vis)
        if self.vis:
            cv2.namedWindow("fix_size",0)
            cv2.imshow("fix_size",img_)

        hm_w = np.expand_dims(hm_w,2)
        if self.vis:
            print("hm.shape : {}".format(hm.shape))
            print("hm_w.shape : {}".format(hm_w.shape))
            print("img_fix_size.shape : {}".format(img_.shape))

        #-------------------------------------
        #-------------------------------------
        if self.flag_agu == True:
            if random.random() > 0.5:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)
        if self.flag_agu == True:
            if random.random() > 0.9:
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img_=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        if self.flag_agu == True:
            if random.random() > 0.95:
                img_ = img_agu_channel_same(img_)

        if self.vis:
            cv2.namedWindow("fix_size_agu",0)
            cv2.imshow("fix_size_agu",img_)
            cv2.waitKey(1)

        img_fix_size = img_.astype(np.float32)


        img_fix_size_r = img_fix_size.astype(np.float32)
        img_fix_size_r = (img_fix_size_r-128.)/256.
        #--------------------------------------------------
        image_fusion = np.concatenate((img_fix_size_r,hm),axis=2)
        if self.vis:
            print(" A image_fusion.shape : {}".format(image_fusion.shape))

        image_fusion = image_fusion.transpose(2, 0, 1)
        if self.vis:
            print(" B image_fusion.shape : {}".format(image_fusion.shape))

        # image_fusion = np.expand_dims(image_fusion,0)
        # if self.vis:
        #     print(" C image_fusion.shape : {}".format(image_fusion.shape))
        #     cv2.waitKey(0)


        gt_3d_joints = np.array(gt_3d_joints).ravel()
        if self.vis:
            print(gt_3d_joints.shape)
            print(image_fusion.shape)
        return image_fusion,gt_3d_joints

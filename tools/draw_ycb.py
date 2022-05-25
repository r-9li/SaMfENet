# 导入
import os
import sys

# Root directory of the project
os.chdir(sys.path[0])
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import cv2

cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
ours_dir = '/media/r/Samsung_USB/ours_result/Densefusion_wo_refine_result'
dataset_config_dir = '/home/r/Dense_ori/DenseFusion-2/datasets/ycb/dataset_config'
posecnn_dir = '/home/r/Dense_ori/DenseFusion-2/experiments/scripts/YCB_Video_toolbox/results_PoseCNN_RSS2018'
densefusion_dir = '/media/r/Samsung_USB/dense_result/Densefusion_wo_refine_result'
dataset_dir = '/home/r/mydisk/YCB/YCB_Video_Dataset'
output_dir = '/home/r/桌面/out_image'
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
cam_mat = np.matrix([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
colors = [(255, 255, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255),
          (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0),
          (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0), (0, 0, 192)]
# load
testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
downsample_rate = 2
rad = 1
tic = 2
render_list = [36, 47, 181, 261, 292, 294, 326, 335, 401, 404, 422, 435, 516, 573, 581, 626, 913, 950, 991, 1128, 1304,
               1477, 1844, 1347, 1458]
# render_list=range(0,2949)
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(dataset_dir, class_input))
    cld[class_id] = []
    downsample_counter = 0
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        downsample_counter += 1
        if downsample_counter % downsample_rate == 0:
            cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

for now in render_list:
    print(now)
    img = Image.open('{0}/{1}-color.png'.format(dataset_dir, testlist[now]))
    img = np.array(img)
    ours_result = scio.loadmat('{0}/{1:04d}.mat'.format(ours_dir, now))
    posecnn_result = scio.loadmat('{0}/{1:06d}.mat'.format(posecnn_dir, now))
    densefusion_result = scio.loadmat('{0}/{1:04d}.mat'.format(densefusion_dir, now))
    gt_result = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_dir, testlist[now]))
    posecnn_rois = np.array(posecnn_result['rois'])
    lst = posecnn_rois[:, 1:2].flatten()
    render_img_ours = img.copy()
    render_img_posecnn = img.copy()
    render_img_posecnnICP = img.copy()
    render_img_densefusion = img.copy()
    render_img_gt = img.copy()
    for idx in range(len(lst)):
        itemid = lst[idx]
        itemid = int(itemid)
        if not any(ours_result['poses'][idx, 0:4]):
            continue

        if [itemid] not in gt_result['cls_indexes'].tolist():
            continue

        ours_r = ours_result['poses'][idx, 0:4]
        ours_t = ours_result['poses'][idx, 4:7]
        ours_r = quaternion_matrix(ours_r)[0:3, 0:3]
        imgpts, _ = cv2.projectPoints(cld[itemid], ours_r, ours_t, cam_mat, dist)
        imgpts = np.floor(imgpts.reshape((-1, 2))).astype(int)
        for px in imgpts:
            cv2.circle(render_img_ours, tuple(px), radius=rad, color=colors[itemid - 1],
                       thickness=tic)

        posecnn_r = posecnn_result['poses'][idx, 0:4]
        posecnn_t = posecnn_result['poses'][idx, 4:7]
        posecnn_r = quaternion_matrix(posecnn_r)[0:3, 0:3]
        imgpts, _ = cv2.projectPoints(cld[itemid], posecnn_r, posecnn_t, cam_mat, dist)
        imgpts = np.floor(imgpts.reshape((-1, 2))).astype(int)
        for px in imgpts:
            cv2.circle(render_img_posecnn, tuple(px), radius=rad, color=colors[itemid - 1],
                       thickness=tic)

        posecnnICP_r = posecnn_result['poses_icp'][idx, 0:4]
        posecnnICP_t = posecnn_result['poses_icp'][idx, 4:7]
        posecnnICP_r = quaternion_matrix(posecnnICP_r)[0:3, 0:3]
        imgpts, _ = cv2.projectPoints(cld[itemid], posecnnICP_r, posecnnICP_t, cam_mat, dist)
        imgpts = np.floor(imgpts.reshape((-1, 2))).astype(int)
        for px in imgpts:
            cv2.circle(render_img_posecnnICP, tuple(px), radius=rad, color=colors[itemid - 1],
                       thickness=tic)

        dense_r = densefusion_result['poses'][idx, 0:4]
        dense_t = densefusion_result['poses'][idx, 4:7]
        dense_r = quaternion_matrix(dense_r)[0:3, 0:3]
        imgpts, _ = cv2.projectPoints(cld[itemid], dense_r, dense_t, cam_mat, dist)
        imgpts = np.floor(imgpts.reshape((-1, 2))).astype(int)
        for px in imgpts:
            cv2.circle(render_img_densefusion, tuple(px), radius=rad,
                       color=colors[itemid - 1],
                       thickness=tic)

        if [itemid] in gt_result['cls_indexes'].tolist():
            gt_idx = gt_result['cls_indexes'].tolist().index([itemid])
            gt_r = gt_result['poses'][0:3, 0:3, gt_idx]
            gt_t = gt_result['poses'][:, 3, gt_idx]
            imgpts, _ = cv2.projectPoints(cld[itemid], gt_r, gt_t, cam_mat, dist)
            imgpts = np.floor(imgpts.reshape((-1, 2))).astype(int)
            for px in imgpts:
                cv2.circle(render_img_gt, tuple(px), radius=rad,
                           color=colors[itemid - 1],
                           thickness=tic)

    render_img_ours = cv2.cvtColor(render_img_ours, cv2.COLOR_BGR2RGB)
    render_img_posecnn = cv2.cvtColor(render_img_posecnn, cv2.COLOR_BGR2RGB)
    render_img_posecnnICP = cv2.cvtColor(render_img_posecnnICP, cv2.COLOR_BGR2RGB)
    render_img_densefusion = cv2.cvtColor(render_img_densefusion, cv2.COLOR_BGR2RGB)
    render_img_gt = cv2.cvtColor(render_img_gt, cv2.COLOR_BGR2RGB)
    cv2.imwrite('{0}/ours/{1:04d}.jpg'.format(output_dir, now), render_img_ours)
    cv2.imwrite('{0}/pose/{1:04d}.jpg'.format(output_dir, now), render_img_posecnn)
    cv2.imwrite('{0}/poseICP/{1:04d}.jpg'.format(output_dir, now), render_img_posecnnICP)
    cv2.imwrite('{0}/dense/{1:04d}.jpg'.format(output_dir, now), render_img_densefusion)
    cv2.imwrite('{0}/gt/{1:04d}.jpg'.format(output_dir, now), render_img_gt)

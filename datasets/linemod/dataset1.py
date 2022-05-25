import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2


class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_cloud = []
        self.list_choose = []
        self.list_img_masked = []
        self.list_target = []
        self.list_model_points = []
        self.edge_img = []
        self.edge_weight = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                if self.mode == 'eval':
                    self.list_label.append(
                        '{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))

                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.safe_load(meta_file)
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))

            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]
        begin_time = time.time()
        print("process begin!")
        print(self.length)
        for index in range(self.length):
            print(index)
            img = Image.open(self.list_rgb[index])
            ori_img = np.array(img)
            depth = np.array(Image.open(self.list_depth[index]))
            label = np.array(Image.open(self.list_label[index]))
            obj = self.list_obj[index]
            rank = self.list_rank[index]
            if obj == 2:
                for i in range(0, len(self.meta[obj][rank])):
                    if self.meta[obj][rank][i]['obj_id'] == 2:
                        meta = self.meta[obj][rank][i]
                        break
            else:
                meta = self.meta[obj][rank][0]
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            if self.mode == 'eval':
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
            else:
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]

            mask = mask_label * mask_depth

            if self.add_noise:
                img = self.trancolor(img)

            img = np.array(img)[:, :, :3]
            edge_img = img
            img = np.transpose(img, (2, 0, 1))
            img_masked = img

            if self.mode == 'eval':
                rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
            else:
                rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

            img_masked = img_masked[:, rmin:rmax, cmin:cmax]
            edge_img = edge_img[rmin:rmax, cmin:cmax, :]
            if self.mode == 'eval':
                label = np.reshape(label, (label.shape[0], label.shape[1], 1))
                label = np.repeat(label, 3, axis=-1)
            label_masked = label[rmin:rmax, cmin:cmax, :]
            bool_label_masked = ma.getmaskarray(ma.masked_equal(label_masked, np.array([255, 255, 255])))
            edge_img_mask = edge_img * bool_label_masked
            edge_img_mask = edge_img_mask.astype(np.uint8)
            edge_img_mask = cv2.cvtColor(edge_img_mask, cv2.COLOR_BGR2GRAY)
            p1 = random.randint(30, 100)
            p2 = random.random() * 0.8 + 1.2
            edge = (cv2.Canny(edge_img_mask, p1, p1 * p2)).astype(np.float32)[:, :, np.newaxis]
            cnt_edge = float(np.count_nonzero(edge))
            weight = np.where(edge == 0, cnt_edge / (edge.shape[0] * edge.shape[1]),
                              1 - cnt_edge / (edge.shape[0] * edge.shape[1]))
            weight = weight.astype(np.float32)
            edge = (edge / 255.).astype(np.float32)
            weight = np.transpose(weight, (2, 0, 1))
            edge = np.transpose(edge, (2, 0, 1))

            # p_img = np.transpose(img_masked, (1, 2, 0))
            # scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

            target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
            target_t = np.array(meta['cam_t_m2c'])
            add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) == 0:
                self.list_cloud.append(np.array([0]))
                self.list_choose.append(np.array([0]))
                self.list_img_masked.append(img_masked.astype(np.float32))
                self.list_target.append(np.array([0]))
                self.list_model_points.append(np.array([0]))
                self.edge_img.append(np.array([0]))
                self.edge_weight.append(np.array([0]))
                continue
            if len(choose) > self.num:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_scale = 1.0
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
            pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)
            cloud = cloud / 1000.0

            if self.add_noise:
                cloud = np.add(cloud, add_t)

            # fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
            # for it in cloud:
            #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            # fw.close()

            model_points = self.pt[obj] / 1000.0
            dellist = [j for j in range(0, len(model_points))]
            dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
            model_points = np.delete(model_points, dellist, axis=0)

            # fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
            # for it in model_points:
            #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            # fw.close()

            target = np.dot(model_points, target_r.T)
            if self.add_noise:
                target = np.add(target, target_t / 1000.0 + add_t)
                out_t = target_t / 1000.0 + add_t
            else:
                target = np.add(target, target_t / 1000.0)
                out_t = target_t / 1000.0

            # fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
            # for it in target:
            #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            # fw.close()
            self.list_cloud.append(cloud.astype(np.float32))
            self.list_choose.append(choose.astype(np.int32))
            self.list_img_masked.append(img_masked.astype(np.float32))
            self.list_target.append(target.astype(np.float32))
            self.list_model_points.append(model_points.astype(np.float32))
            self.edge_img.append(edge)
            self.edge_weight.append(weight)
        print("process down!")
        print(time.time() - begin_time)

    def __getitem__(self, index):
        obj = self.list_obj[index]
        cloud = self.list_cloud[index]
        choose = self.list_choose[index]
        img_masked = self.list_img_masked[index]
        target = self.list_target[index]
        model_points = self.list_model_points[index]
        edge = self.edge_img[index]
        edge_weight = self.edge_weight[index]
        return torch.from_numpy(cloud), \
               torch.LongTensor(choose), \
               self.norm(torch.from_numpy(img_masked)), \
               torch.from_numpy(target), \
               torch.from_numpy(model_points), \
               torch.LongTensor([self.objlist.index(obj)]), \
               torch.from_numpy(edge), \
               torch.from_numpy(edge_weight)

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

import os
import time
import argparse
import random
import cv2
import copy
import _pickle as cPickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from lib.network import StereoPoseNet, StereoPoseNet_with_depth
from lib.utils import get_bbox
from lib.utils import get_3d_bbox, transform_coordinates_3d, calculate_2d_projections, draw_bboxes, draw_axis
from lib.align import *

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='one_door_cabinet')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=224, help='cropped image size')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train')
parser.add_argument('--resume_epoch', type=int, default=None, help='resume from saved model')
parser.add_argument('--img_arc', type=str, default='ResNet18', help='ResNet18, ResNet34')
parser.add_argument('--result_root_dir', type=str, default='results/inference', help='directory to save train results')
parser.add_argument('--exp_name', type=str, default=None, help='name used to distinguish different experiments')

parser.add_argument('--save_vis', action='store_true', default=False)
opt = parser.parse_args()
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
def show_image(img) :

    mat = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if len(img.shape) == 2:
        mat[:, :, 0] = img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        mat = img
    else :
        raise ValueError("Invalid image shape")
    
    cv2.imshow("tmp", mat)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def draw_result(img, nocs, scale, r, t, k):
    half_size = np.max(abs(nocs), axis=0)
    size = 2 * half_size * scale
    bbox = get_3d_bbox(size)

    sRT = np.eye(4).astype(np.float32)
    sRT[:3, :3] = r
    sRT[:3, 3] = t.flatten()

    transformed_bbox_3d = transform_coordinates_3d(bbox, sRT)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, k)
    img = draw_bboxes(img, projected_bbox, (0, 0, 255))

    axis_endopoints = np.array([[0.0, 0.02, 0.0, 0.0], [0.0, 0.0, 0.02, 0.0], [0.0, 0.0, 0.0, 0.02]])
    transformed_endpoints = transform_coordinates_3d(axis_endopoints, sRT)
    projected_endpoints = calculate_2d_projections(transformed_endpoints, k)
    img = draw_axis(img, projected_endpoints)

    return img, transformed_bbox_3d

def load_single_view_data(view_path):
    rgb = cv2.imread(view_path + '_color.png')[:, :, :3]
    rgb = rgb[:, :, ::-1]

    with open(view_path + '_label.pkl', 'rb') as f:
        view_label = cPickle.load(f)
    f.close()

    view_intrinsic = view_label['camera_intrinsic']
    view_extrinsic = view_label['camera_extrinsic']
    with open(view_path + '_masks.pkl', 'rb') as f:
        view_masks = cPickle.load(f)
    f.close()

    return rgb, view_label, view_masks, view_intrinsic, view_extrinsic

def prepare_model_input(rgb, masks, target_part_name, intrinsic, resize_size = opt.img_size):
    mask = masks[target_part_name]
    ys, xs = np.nonzero(mask)
    y1 = np.min(ys)
    y2 = np.max(ys)
    x1 = np.min(xs)
    x2 = np.max(xs)
    rmin, rmax, cmin, cmax = get_bbox([y1, x1, y2, x2])

    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) > 1024:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:1024] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, 1024-len(choose)), 'wrap')
    
    xmap = np.array([[i for i in range(rgb.shape[1])] for j in range(rgb.shape[0])])
    ymap = np.array([[j for i in range(rgb.shape[1])] for j in range(rgb.shape[0])])

    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    view_pts2d = np.concatenate((xmap_masked, ymap_masked), axis=-1)

    view_rgb = rgb[rmin:rmax, cmin:cmax, :]
    view_rgb = cv2.resize(view_rgb, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    crop_w = rmax - rmin
    ratio = resize_size / crop_w
    col_idx = choose % crop_w
    row_idx = choose // crop_w
    choose = (np.floor(row_idx * ratio) * resize_size + np.floor(col_idx * ratio)).astype(np.int64)
    view_rgb = transform(view_rgb)

    # adjust the intrinsic matrix
    # corping and resize will change the intrinsic matrix
    cam_fx = intrinsic[0, 0]
    cam_fy = intrinsic[1, 1]
    cam_cx = intrinsic[0, 2]
    cam_cy = intrinsic[1, 2]
    crop_center = [float(cmin + cmax) / 2, float(rmin + rmax) / 2]
    crop_size = [float(cmax - cmin + 1), float(rmax - rmin + 1)]
    camera_cx = (cam_cx - (crop_center[0] - crop_size[0] / 2)) * ratio
    camera_cy = (cam_cy - (crop_center[1] - crop_size[1] / 2)) * ratio
    camera_fx = cam_fx * ratio
    camera_fy = cam_fy * ratio

    new_intrinsic = np.eye(3)
    new_intrinsic[0, 0] = camera_fx
    new_intrinsic[1, 1] = camera_fy
    new_intrinsic[0, 2] = camera_cx
    new_intrinsic[1, 2] = camera_cy

    return view_rgb, choose, view_pts2d, new_intrinsic

def test():
    estimator = StereoPoseNet_with_depth(n_cat=1, nv_pts=opt.n_pts)
    estimator = nn.DataParallel(estimator)
    estimator.cuda()

    opt.result_dir = os.path.join(opt.result_root_dir, opt.task_name)

    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)

    vis_dir = os.path.join(opt.result_dir, 'vis')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    checkpoint_path = 'checkpoint/with_depth/v_model.pth'
    estimator.load_state_dict(torch.load(checkpoint_path))
    
    test_list = os.path.join(opt.data_dir, opt.task_name + '_test_list.txt')
    img_list = [line.rstrip('\n') for line in open(test_list)]
    
    for i in range(len(img_list)):
        print(img_list[i])

        view1_path = os.path.join(opt.data_dir, img_list[i].split(',')[0])
        view2_path = os.path.join(opt.data_dir, img_list[i].split(',')[1])

        rgb1, view1_label, view1_masks, view1_intrinsic, view1_extrinsic = load_single_view_data(view1_path)
        rgb2, view2_label, view2_masks, view2_intrinsic, view2_extrinsic = load_single_view_data(view2_path)
        overlap_part_list = [part for part in view1_label['handle_list'] if part in view2_label['handle_list']]

        for j in range(len(overlap_part_list)):
            target_part_name = overlap_part_list[j]
            view1_rgb, view1_choose, view1_pts2d, new_view1_intrinsic = prepare_model_input(rgb1, view1_masks, target_part_name, view1_intrinsic)
            view2_rgb, view2_choose, view2_pts2d, new_view2_intrinsic = prepare_model_input(rgb2, view2_masks, target_part_name, view2_intrinsic)

            view1_rgb = view1_rgb.unsqueeze(0).cuda()
            view1_choose = torch.from_numpy(view1_choose).unsqueeze(0).cuda()
            view2_rgb = view2_rgb.unsqueeze(0).cuda()
            view2_choose = torch.from_numpy(view2_choose).unsqueeze(0).cuda()

            P1 = np.eye(4)
            P1[:3, :] = new_view1_intrinsic @ view1_extrinsic[:3, :]
            P2 = np.eye(4)
            P2[:3, :] = new_view2_intrinsic @ view2_extrinsic[:3, :]

            P1 = torch.from_numpy(P1).unsqueeze(0).cuda().float()
            P2 = torch.from_numpy(P2).unsqueeze(0).cuda().float()

            depth_min = 0.1
            depth_interval = 0.1
            n_depths = 24
            depth_values = np.arange(depth_min, depth_interval * (n_depths - 0.5) + depth_min, depth_interval,
                                                dtype=np.float32)
            depth_values = torch.from_numpy(depth_values).unsqueeze(0).cuda().float()

            with torch.no_grad():
                prediction = estimator(view1_rgb, view1_choose, view2_rgb, view2_choose, P1, P2, depth_values)
            
            pred_view1_nocs = prediction['view1_nocs'][0].cpu().numpy()
            pred_view2_nocs = prediction['view2_nocs'][0].cpu().numpy()

            pred_view1_depth = prediction['view1_depth'][0].cpu().numpy()
            pred_view2_depth = prediction['view2_depth'][0].cpu().numpy()

            view1_choose = view1_choose[0].cpu().numpy()
            view1_depth_masked = pred_view1_depth.flatten()[view1_choose][:, np.newaxis]
            xmap = np.array([[i for i in range(opt.img_size)] for j in range(opt.img_size)])
            ymap = np.array([[j for i in range(opt.img_size)] for j in range(opt.img_size)])

            view1_xmap_masked = xmap.flatten()[view1_choose][:, np.newaxis]
            view1_ymap_masked = ymap.flatten()[view1_choose][:, np.newaxis]
            cam_cx = new_view1_intrinsic[0, 2]
            cam_cy = new_view1_intrinsic[1, 2]
            cam_fx = new_view1_intrinsic[0, 0]
            cam_fy = new_view1_intrinsic[1, 1]
            pt2 = view1_depth_masked
            pt0 = (view1_xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (view1_ymap_masked - cam_cy) * pt2 / cam_fy
            view1_points = np.concatenate((pt0, pt1, pt2), axis=1)

            ts, tr, tt, tsRT = estimateSimilarityTransform(pred_view1_nocs, view1_points)
            
            # compute bbox
            half_size = np.max(abs(pred_view1_nocs), axis=0)
            size = 2 * half_size * ts
            bbox = get_3d_bbox(size)

            sRT = np.eye(4).astype(np.float32)
            sRT[:3, :3] = tr
            sRT[:3, 3] = tt.flatten()
            bbox = transform_coordinates_3d(bbox, sRT)
            
            if opt.save_vis:
                dis_img = copy.deepcopy(rgb1[:, :, ::-1])

                dis, _ = draw_result(dis_img, pred_view1_nocs, ts, tr, tt, view1_intrinsic)
                cv2.imwrite(os.path.join(vis_dir, str(i).zfill(4) + '_' + str(j).zfill(2) + '.png'), dis)

if __name__ == '__main__':
    test()

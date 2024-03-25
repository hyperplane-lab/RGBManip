import os
import argparse
import cv2
import copy
import _pickle as cPickle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.pose_estimator.base_estimator import BasePoseEstimator
from env.sapien_envs.open_cabinet import OpenCabinetEnv
from utils.tools import show_image

from .lib.network_v5 import StereoPoseNet, StereoPoseNet_with_depth
from .lib.utils import get_bbox, depth_estimation_from_nocs_matches, compute_scale_and_translation
from .lib.utils import get_3d_bbox, transform_coordinates_3d, calculate_2d_projections, draw_bboxes
from .lib.align import *

def draw_result(img, nocs, scale, r, t, k):
    img = copy.deepcopy(img)
    half_size = np.max(abs(nocs), axis=0)
    size = 2 * half_size * scale
    bbox = get_3d_bbox(size)

    sRT = np.eye(4).astype(np.float32)
    sRT[:3, :3] = r
    sRT[:3, 3] = t.flatten()

    transformed_bbox_3d = transform_coordinates_3d(bbox, sRT)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, k)
    img = draw_bboxes(img, projected_bbox, (0, 0, 1))

    return img, transformed_bbox_3d

class AdaPoseEstimator_v5(BasePoseEstimator) :

    def __init__(self, env, cfg, logger) :

        super().__init__(env, cfg, logger)

        self.estimator = StereoPoseNet_with_depth(
            n_cat=1,
            nv_pts=cfg["n_pts"],
            regress_pose=cfg["direct_regression"]
        )
        self.estimator = nn.DataParallel(self.estimator)
        self.estimator.cuda()
        self.cfg = cfg

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        if self.cfg["load"] :
            self.estimator.load_state_dict(torch.load(self.cfg["checkpoint_path"]))

    def prepare_model_input(self, rgb, mask, intrinsic, resize_size):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None, None, None, None
        else :
            y1 = np.min(ys, initial=rgb.shape[0])
            y2 = np.max(ys, initial=0)
        if len(xs) == 0:
            return None, None, None, None
        else :
            x1 = np.min(xs, initial=rgb.shape[1])
            x2 = np.max(xs, initial=0)
        rmin, rmax, cmin, cmax = get_bbox([y1, x1, y2, x2])

        # oldversion: sample->resize
        # choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        # if len(choose) > 1024:
        #     c_mask = np.zeros(len(choose), dtype=int)
        #     c_mask[:1024] = 1
        #     np.random.shuffle(c_mask)
        #     choose = choose[c_mask.nonzero()]
        # elif len(choose) == 0:
        #     return None, None, None, None
        # else:
        #     choose = np.pad(choose, (0, 1024-len(choose)), 'wrap')
        
        # xmap = np.array([[i for i in range(rgb.shape[1])] for j in range(rgb.shape[0])])
        # ymap = np.array([[j for i in range(rgb.shape[1])] for j in range(rgb.shape[0])])

        # xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        # ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        # view_pts2d = np.concatenate((xmap_masked, ymap_masked), axis=-1)

        # view_rgb = rgb[rmin:rmax, cmin:cmax, :]
        # view_rgb = cv2.resize(view_rgb, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
        # crop_w = rmax - rmin
        # ratio = resize_size / crop_w
        # col_idx = choose % crop_w
        # row_idx = choose // crop_w
        # choose = (np.floor(row_idx * ratio) * resize_size + np.floor(col_idx * ratio)).astype(np.int64)
        # view_rgb = self.transform(view_rgb)

        # # adjust the intrinsic matrix
        # # corping and resize will change the intrinsic matrix
        # cam_fx = intrinsic[0, 0]
        # cam_fy = intrinsic[1, 1]
        # cam_cx = intrinsic[0, 2]
        # cam_cy = intrinsic[1, 2]
        # crop_center = [float(cmin + cmax) / 2, float(rmin + rmax) / 2]
        # crop_size = [float(cmax - cmin + 1), float(rmax - rmin + 1)]
        # camera_cx = (cam_cx - (crop_center[0] - crop_size[0] / 2)) * ratio
        # camera_cy = (cam_cy - (crop_center[1] - crop_size[1] / 2)) * ratio
        # camera_fx = cam_fx * ratio
        # camera_fy = cam_fy * ratio

        # new_intrinsic = np.eye(3)
        # new_intrinsic[0, 0] = camera_fx
        # new_intrinsic[1, 1] = camera_fy
        # new_intrinsic[0, 2] = camera_cx
        # new_intrinsic[1, 2] = camera_cy
        
        # return view_rgb, choose, view_pts2d, new_intrinsic

        # New version: resize->sample
        original_mask = mask[rmin:rmax, cmin:cmax]
        resize_mask = cv2.resize(original_mask.astype(np.float32), (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
        choose = resize_mask.flatten().nonzero()[0]

        if len(choose) > 1024:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:1024] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif len(choose) == 0:
            return None, None, None, None
        else:
            choose = np.pad(choose, (0, 1024-len(choose)), 'wrap')
        
        crop_w = rmax - rmin
        ratio = resize_size / crop_w
        xmap = np.array([[i for i in range(resize_size)] for j in range(resize_size)])
        ymap = np.array([[j for i in range(resize_size)] for j in range(resize_size)])
        xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        xmap_masked = xmap_masked / ratio + cmin
        ymap_masked = ymap_masked / ratio + rmin
        view_pts2d = np.concatenate((xmap_masked, ymap_masked), axis=-1)

        view_rgb = rgb[rmin:rmax, cmin:cmax, :]
        view_rgb = cv2.resize(view_rgb, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
        view_rgb = self.transform(view_rgb)

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

    # def prepare_model_input(self, rgb, mask, resize_size):
    #     ys, xs = np.nonzero(mask)
    #     if len(ys) == 0:
    #         return None, None, None
    #     else :
    #         y1 = np.min(ys, initial=rgb.shape[0])
    #         y2 = np.max(ys, initial=0)
    #     if len(xs) == 0:
    #         return None, None, None
    #     else :
    #         x1 = np.min(xs, initial=rgb.shape[1])
    #         x2 = np.max(xs, initial=0)
    #     rmin, rmax, cmin, cmax = get_bbox([y1, x1, y2, x2])

    #     choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    #     if len(choose) > 1024:
    #         c_mask = np.zeros(len(choose), dtype=int)
    #         c_mask[:1024] = 1
    #         np.random.shuffle(c_mask)
    #         choose = choose[c_mask.nonzero()]
    #     else:
    #         choose = np.pad(choose, (0, 1024-len(choose)), 'wrap')
        
    #     xmap = np.array([[i for i in range(rgb.shape[1])] for j in range(rgb.shape[0])])
    #     ymap = np.array([[j for i in range(rgb.shape[1])] for j in range(rgb.shape[0])])

    #     xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    #     ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    #     view_pts2d = np.concatenate((xmap_masked, ymap_masked), axis=-1)

    #     view_rgb = rgb[rmin:rmax, cmin:cmax, :]
    #     view_rgb = cv2.resize(view_rgb, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    #     crop_w = rmax - rmin
    #     ratio = resize_size / crop_w
    #     col_idx = choose % crop_w
    #     row_idx = choose // crop_w
    #     choose = (np.floor(row_idx * ratio) * resize_size + np.floor(col_idx * ratio)).astype(np.int64)
    #     view_rgb = self.transform(view_rgb)

    #     return view_rgb, choose, view_pts2d
    
    def estimate(self, camera_intrinsic_batch, rgb1_batch, view1_mask_batch, view1_extrinsic_batch, \
                rgb2_batch, view2_mask_batch, view2_extrinsic_batch) :

        bbox_list = []

        for camera_intrinsic, rgb1, view1_mask, view1_extrinsic, \
            rgb2, view2_mask, view2_extrinsic in zip(camera_intrinsic_batch, rgb1_batch, view1_mask_batch, view1_extrinsic_batch, \
                rgb2_batch, view2_mask_batch, view2_extrinsic_batch) :
        
            bbox = self.predict(camera_intrinsic, rgb1, view1_mask, view1_extrinsic, \
                         rgb2, view2_mask, view2_extrinsic)

            bbox_list.append(bbox)
        
        return np.asarray(bbox_list)

    def predict(self, camera_intrinsic, rgb1, view1_mask, view1_extrinsic, \
                rgb2, view2_mask, view2_extrinsic) :
    
        default_bbox = np.asarray([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]) + 10.0
        
        view1_rgb, view1_choose, view1_pts2d, new_view1_intrinsic = self.prepare_model_input(
            rgb1,
            view1_mask,
            camera_intrinsic,
            resize_size=self.cfg["img_size"]
        )
        view2_rgb, view2_choose, view2_pts2d, new_view2_intrinsic = self.prepare_model_input(
            rgb2,
            view2_mask,
            camera_intrinsic,
            resize_size=self.cfg["img_size"]
        )

        if view1_rgb == None or view2_rgb == None:
            return default_bbox

        view1_rgb = view1_rgb.unsqueeze(0).cuda().float()
        view1_choose = torch.from_numpy(view1_choose).unsqueeze(0).cuda()
        view2_rgb = view2_rgb.unsqueeze(0).cuda().float()
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
            prediction = self.estimator(view1_rgb, view1_choose, view2_rgb, view2_choose, P1, P2, depth_values)
        
        pred_view1_nocs = prediction['view1_nocs'][0].cpu().numpy()
        pred_view2_nocs = prediction['view2_nocs'][0].cpu().numpy()

        pred_view1_depth = prediction['view1_depth'][0].cpu().numpy()
        pred_view2_depth = prediction['view2_depth'][0].cpu().numpy()

        # Old versoin
        # view1_choose = view1_choose[0].cpu().numpy()
        # view1_depth_masked = pred_view1_depth.flatten()[view1_choose][:, np.newaxis]
        # xmap = np.array([[i for i in range(self.cfg["img_size"])] for j in range(self.cfg["img_size"])])
        # ymap = np.array([[j for i in range(self.cfg["img_size"])] for j in range(self.cfg["img_size"])])

        # view1_xmap_masked = xmap.flatten()[view1_choose][:, np.newaxis]
        # view1_ymap_masked = ymap.flatten()[view1_choose][:, np.newaxis]
        # cam_cx = new_view1_intrinsic[0, 2]
        # cam_cy = new_view1_intrinsic[1, 2]
        # cam_fx = new_view1_intrinsic[0, 0]
        # cam_fy = new_view1_intrinsic[1, 1]
        # pt2 = view1_depth_masked
        # pt0 = (view1_xmap_masked - cam_cx) * pt2 / cam_fx
        # pt1 = (view1_ymap_masked - cam_cy) * pt2 / cam_fy
        # view1_points = np.concatenate((pt0, pt1, pt2), axis=1)

        # ts, tr, tt, tsRT = estimateSimilarityTransform(pred_view1_nocs, view1_points)

        # camera_intrinsic, rgb1, view1_mask, view1_extrinsic, \
        #     rgb2, view2_mask, view2_extrinsic

        # if ts == None :
        #     return default_bbox
        
        # # compute bbox
        # half_size = np.max(abs(pred_view1_nocs), axis=0)
        # size = 2 * half_size * ts
        # bbox = get_3d_bbox(size)

        if self.cfg["direct_regression"] :
            tr = prediction['view1_r'][0].cpu().numpy()
            tt, ts = compute_scale_and_translation(pred_view1_depth, pred_view1_nocs, \
                view1_choose[0].cpu().numpy(), new_view1_intrinsic, self.cfg["img_size"], tr)
        elif self.cfg["use_depth"]:
            view1_choose = view1_choose[0].cpu().numpy()
            xmap = np.array([[i for i in range(self.cfg["img_size"])] for j in range(self.cfg["img_size"])])
            ymap = np.array([[j for i in range(self.cfg["img_size"])] for j in range(self.cfg["img_size"])])

            view1_xmap_masked = xmap.flatten()[view1_choose][:, np.newaxis]
            view1_ymap_masked = ymap.flatten()[view1_choose][:, np.newaxis]
            cam_cx = new_view1_intrinsic[0, 2]
            cam_cy = new_view1_intrinsic[1, 2]
            cam_fx = new_view1_intrinsic[0, 0]
            cam_fy = new_view1_intrinsic[1, 1]
            pt2 = pred_view1_depth.flatten()[:, np.newaxis]
            pt0 = (view1_xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (view1_ymap_masked - cam_cy) * pt2 / cam_fy
            view1_points = np.concatenate((pt0, pt1, pt2), axis=1)

            ts, tr, tt, tsRT = estimateSimilarityTransform(pred_view1_nocs, view1_points)
        else:
            P1 = np.eye(4)
            P1[:3, :] = camera_intrinsic @ view1_extrinsic[:3, :]
            P2 = np.eye(4)
            P2[:3, :] = camera_intrinsic @ view2_extrinsic[:3, :]

            results = depth_estimation_from_nocs_matches(view1_pts2d, pred_view1_nocs, \
                P1, view1_extrinsic, view2_pts2d, pred_view2_nocs, P2, view2_extrinsic, camera_intrinsic)
            
            _, ts, tr, tt, tsRT = estimatePnPRansac(pred_view1_nocs.astype(np.float32), \
                view1_pts2d.astype(np.float32), results[0], camera_intrinsic)
        
        if ts == None :
            return default_bbox
        
        # compute bbox
        half_size = np.max(abs(pred_view1_nocs), axis=0)
        size = 2 * half_size * ts
        bbox = get_3d_bbox(size)

        sRT = np.eye(4).astype(np.float32)
        sRT[:3, :3] = tr
        sRT[:3, 3] = tt.flatten()
        bbox = transform_coordinates_3d(bbox, sRT)

        dis1, _ = draw_result(rgb1[:, :, ::-1], pred_view1_nocs, ts, tr, tt, camera_intrinsic)

        # show_image(dis1)

        # Project to world frame
        ex_inv = np.linalg.inv(view1_extrinsic)

        if np.isfinite(ex_inv).all() and np.isfinite(bbox).all():
            return (ex_inv[:3, :3] @ bbox + ex_inv[:3, 3:4]).T
        else :
            return default_bbox
        ######

        '''
        Old version
        '''
        
        view1_rgb, view1_choose, view1_pts2d = self.prepare_model_input(rgb1, view1_mask, resize_size = self.cfg["img_size"])
        view2_rgb, view2_choose, view2_pts2d = self.prepare_model_input(rgb2, view2_mask, resize_size = self.cfg["img_size"])

        if view1_rgb == None or view2_rgb == None:
            return np.zeros((8, 3))

        view1_rgb = view1_rgb.unsqueeze(0).cuda().float()
        view1_choose = torch.from_numpy(view1_choose).unsqueeze(0).cuda().long()
        view2_rgb = view2_rgb.unsqueeze(0).cuda().float()
        view2_choose = torch.from_numpy(view2_choose).unsqueeze(0).cuda().long()

        with torch.no_grad():
            prediction = self.estimator(view1_rgb, view1_choose, view2_rgb, view2_choose)

            pred_view1_nocs = prediction['view1_nocs'][0].cpu().numpy()
            pred_view2_nocs = prediction['view2_nocs'][0].cpu().numpy()

            P1 = camera_intrinsic @ view1_extrinsic[:3, :]
            P2 = camera_intrinsic @ view2_extrinsic[:3, :]

            # two views have the same intrinsic
            results = depth_estimation_from_nocs_matches(view1_pts2d, pred_view1_nocs, \
                P1, view2_pts2d, pred_view2_nocs, P2, camera_intrinsic)

            _, ts, tr, tt, tsRT = estimatePnPRansac(pred_view1_nocs.astype(np.float32), \
                view1_pts2d.astype(np.float32), results[0], camera_intrinsic)
            
            # compute bbox
            half_size = np.max(abs(pred_view1_nocs), axis=0)
            size = 2 * half_size * ts
            bbox = get_3d_bbox(size)

            sRT = np.eye(4).astype(np.float32)
            sRT[:3, :3] = tr
            sRT[:3, 3] = tt.flatten()
            bbox = transform_coordinates_3d(bbox, sRT)

            dis, _ = draw_result(rgb1[:, :, ::-1], pred_view1_nocs, ts, tr, tt, camera_intrinsic)

            # show_image(dis)
            # cv2.imwrite("miscs/estimation.jpg", dis*255)

            # Project to world frame
            ex_inv = np.linalg.inv(view1_extrinsic)
            return (ex_inv[:3, :3] @ bbox + ex_inv[:3, 3:4]).T

if __name__ == "__main__" :

    pass
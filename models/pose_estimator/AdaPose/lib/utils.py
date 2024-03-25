import logging
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
from tqdm import tqdm

def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

def compute_scale(cam_pts_3d, nocs_pts):
    real_dis = cam_pts_3d[:, np.newaxis, :] - cam_pts_3d[np.newaxis, :, :]
    real_dis = np.linalg.norm(real_dis, axis=-1).flatten()
    nocs_dis = nocs_pts[:, np.newaxis, :] - nocs_pts[np.newaxis, :, :]
    nocs_dis = np.linalg.norm(nocs_dis, axis=-1).flatten()

    dis_id = np.arange(nocs_dis.shape[0])
    valid_dis_flag = (nocs_dis > 0.01)
    valid_dis_id = dis_id[valid_dis_flag]
    
    valid_real_dis = real_dis[valid_dis_id]
    valid_nocs_dis = nocs_dis[valid_dis_id]

    valid_dis_flag = (valid_real_dis < 0.3)
    valid_dis_id = valid_dis_id[valid_dis_flag]

    valid_real_dis = real_dis[valid_dis_id]
    valid_nocs_dis = nocs_dis[valid_dis_id]

    scales = valid_real_dis / valid_nocs_dis
    return np.median(scales)

def compute_scale_and_translation(pred_depth, pred_nocs, choose, intrinsic, img_size, rotation):
    xmap = np.array([[i for i in range(img_size)] for j in range(img_size)])
    ymap = np.array([[j for i in range(img_size)] for j in range(img_size)])

    xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    ymap_masked = ymap.flatten()[choose][:, np.newaxis]
    cam_cx = intrinsic[0, 2]
    cam_cy = intrinsic[1, 2]
    cam_fx = intrinsic[0, 0]
    cam_fy = intrinsic[1, 1]
    pt2 = pred_depth[:, np.newaxis]
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    camera_points = np.concatenate((pt0, pt1, pt2), axis=1)
    scale = compute_scale(camera_points, pred_nocs)

    RT = np.eye(4)
    RT[:3, :3] = scale*rotation
    temp_pts = transform_coordinates_3d(pred_nocs.T, RT).T

    translation = np.mean(camera_points, axis=0) - np.mean(temp_pts, axis=0)
    return translation, scale

def depth_estimation_from_nocs_matches(left_camPts, left_nocs, left_proj, left_pose, right_camPts, right_nocs, right_proj, right_pose, intrinsic):
    # nocs_distance_threshold = 0.10

    # distance matrix between left and right nocs
    dis = left_nocs[:, np.newaxis, :] - right_nocs[np.newaxis, :, :]
    dis = np.linalg.norm(dis, axis=-1)

    matched_id_left_to_right = np.argmin(dis, axis=1)
    matched_id_right_to_left = np.argmin(dis, axis=0)

    left_id = np.arange(left_nocs.shape[0])

    # cross check to extract the mutually matched points in nocs space
    check_id = matched_id_right_to_left[matched_id_left_to_right]
    matched_flag = (check_id == left_id)
    matched_left_id = left_id[matched_flag]
    matched_right_id = matched_id_left_to_right[matched_left_id]
    print('number of matches after stage1:', len(matched_left_id))

    match_dis = dis[matched_left_id, matched_right_id]
    matched_flag = (match_dis < 0.01)
    matched_left_id = matched_left_id[matched_flag]
    matched_right_id = matched_right_id[matched_flag]
    print('number of matches after stage2:', len(matched_left_id))

    relative_extrinsic = left_pose @ np.linalg.inv(right_pose)
    relative_r1 = relative_extrinsic[:3, :3]
    relative_t1 = relative_extrinsic[:3, 3]

    tx = np.zeros((3, 3)).astype(np.float32)
    tx[0, 1] = -relative_t1[2]
    tx[1, 0] = relative_t1[2]
    tx[0, 2] = relative_t1[1]
    tx[2, 0] = -relative_t1[1]
    tx[1, 2] = -relative_t1[0]
    tx[2, 1] = relative_t1[0]
    f21 = (np.linalg.inv(intrinsic).T) @ tx @ \
        relative_r1 @ (np.linalg.inv(intrinsic))

    left_matched_pts2d = left_camPts[matched_left_id, :]
    right_matched_pts2d = right_camPts[matched_right_id, :]
    left_camPts_3D = np.ones((3, left_matched_pts2d.shape[0]))
    right_camPts_3D = np.ones((3, right_matched_pts2d.shape[0]))
    left_camPts_3D[0], left_camPts_3D[1] = left_matched_pts2d[:, 0].copy(), left_matched_pts2d[:, 1].copy()
    right_camPts_3D[0], right_camPts_3D[1] = right_matched_pts2d[:, 0].copy(), right_matched_pts2d[:, 1].copy()

    epipolar_dis = (left_camPts_3D.T) @ f21 @ right_camPts_3D
    epipolar_dis = np.abs(np.diagonal(epipolar_dis))

    matched_flag = (epipolar_dis < 1.0)
    matched_left_id = matched_left_id[matched_flag]
    matched_right_id = matched_right_id[matched_flag]
    print('number of matches after stage3:', len(matched_left_id))

    # pick out matched pts on imgae
    left_matched_pts2d = left_camPts[matched_left_id, :]
    left_matched_nocs = left_nocs[matched_left_id, :]

    right_matched_pts2d = right_camPts[matched_right_id, :]
    right_matched_nocs = right_nocs[matched_right_id, :]

    left_camPts_3D = np.ones((3, left_matched_pts2d.shape[0]))
    right_camPts_3D = np.ones((3, right_matched_pts2d.shape[0]))
    left_camPts_3D[0], left_camPts_3D[1] = left_matched_pts2d[:, 0].copy(), left_matched_pts2d[:, 1].copy()
    right_camPts_3D[0], right_camPts_3D[1] = right_matched_pts2d[:, 0].copy(), right_matched_pts2d[:, 1].copy()
    X = cv2.triangulatePoints(left_proj[:3], right_proj[:3], left_camPts_3D[:2], right_camPts_3D[:2])
    X /= X[3]

    left_Pts = left_pose @ X
    right_Pts = right_pose @ X

    left_scale = compute_scale(left_Pts[:3].T, left_matched_nocs)
    right_scale = compute_scale(right_Pts[:3].T, right_matched_nocs)

    return left_scale, right_scale, left_matched_pts2d, right_matched_pts2d

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def draw_axis(img, img_pts):
    img_pts = np.int32(img_pts).reshape(-1, 2)

    # draw the x-axis
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[1]), (0, 0, 255), 3)

    # draw the y-axis
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[2]), (0, 255, 0), 3)

    # draw the z-axis
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[3]), (255, 0, 0), 3)

    return img

def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img
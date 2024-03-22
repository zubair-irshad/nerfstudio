import numpy as np
from plyfile import PlyData, PlyElement
from nerfstudio.utils.io import load_from_json
from pathlib import Path
from nerfstudio.cameras import camera_utils
import torch
import os

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    return xyz
    

def get_scaled_bounding_boxes(data_path):

    split = "train"
    path = data_path / f"transforms_{split}.json"
    meta = load_from_json(path)
    image_filenames = []
    poses = []
    for frame in meta["frames"]:
        
        if frame["file_path"].startswith("./"):
            fname = data_path / Path(frame["file_path"].replace("./", "") + ".png")
        else:
            fname = data_path / Path(frame["file_path"])
        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))

    poses = np.array(poses).astype(np.float32)
    camera_to_worlds = torch.from_numpy(poses)  # camera to world transform
    camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
        camera_to_worlds,
        method="none",
        center_method="poses",
    )

    all_rotation = []
    all_translation = []
    all_extents = []
    for box in meta["bounding_boxes"]:
        print(box)

        R = torch.from_numpy(np.array(box["orientation"])).reshape(3,3)
        T = torch.from_numpy(np.array(box["position"])).reshape(3)
        TM = torch.eye(4)
        TM[:3, :3] = R
        TM[:3, 3] = T
        new_T = transform @ TM

        all_rotation.append(new_T[:3,:3].numpy())
        all_translation.append(new_T[:3,3].numpy())
        all_extents.append(np.array(box["extents"]))

    return all_rotation, all_translation, all_extents   




ply_data_path = '/home/zubairirshad/Downloads/point_cloud.cleaned.ply'
# ply_data_path = '/home/zubairirshad/nerfstudio/outputs/gs_front3d_colmap_pts_center_origin/gaussian-splatting/2024-01-20_122452/point_cloud.ply'

xyz = load_ply(ply_data_path)

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# o3d.visualization.draw_geometries([pcd])

path = '/home/zubairirshad/Downloads/front3d_single_scene/3dfront_0022_01/train'
path = Path(path)
all_rotation, all_translation, all_extents = get_scaled_bounding_boxes(path)

#Now visualize all the bounding boxes as oriented boxes

all_oriented_boxes = []
for rotation, translation, extent in zip(all_rotation, all_translation, all_extents):
    print(rotation, translation, extent)
    # print(rotation.shape, translation.shape, extent.sh
    oriented_box = o3d.geometry.OrientedBoundingBox(translation, rotation, extent)
    all_oriented_boxes.append(oriented_box)

o3d.visualization.draw_geometries([pcd, *all_oriented_boxes])



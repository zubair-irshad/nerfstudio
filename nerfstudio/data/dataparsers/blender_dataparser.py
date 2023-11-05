# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Blender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    load_3d_points: bool = False
    """Whether to load 3D points from the dataset."""
    masks_path: Optional[Path] = None
    """Path to mask file. If not None, will load mask from this file."""
    features_path: Optional[Path] = None
    """Path to features file. If not None, will load features from this file."""


@dataclass
class Blender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderDataParserConfig

    def __init__(self, config: BlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _load_3D_points(self):
        num_pts = 1000000
        min_bound = np.array([-4, -4, -4])
        max_bound = np.array([4, 4, 4])
        # num_pts = 10  # Change this to the desired number of random points

        # Generate random points within the specified bounds
        # xyz = np.random.rand(num_pts, 3)  # Generates random values between 0 and 1
        # xyz = xyz * (max_bound - min_bound) + min_bound

        xyz = np.random.uniform(min_bound, max_bound, size=(num_pts, 3))

        rgb = np.random.random((num_pts, 3)) * 255.0
        xyz = torch.from_numpy(xyz).float()
        rgb = torch.from_numpy(rgb).float()
        out = {
            "points3D_xyz": xyz,
            "points3D_rgb": rgb,
        }
        return out

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        mask_filenames = []
        features_filenames = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if self.config.masks_path is not None:
                mask_filenames.append(self.config.masks_path / Path(frame["file_path"] + ".png"))

            if self.config.features_path is not None:
                features_filenames.append(self.config.features_path / Path(frame["file_path"] + ".npy.npz"))

        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]

        print("image_height, image_width", image_height, image_width)
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0

        # cx = (image_width - 1.0) / 2.0
        # cy = (image_height - 1.0) / 2.0

        camera_to_worlds = torch.from_numpy(poses)  # camera to world transform

        camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
            camera_to_worlds,
            method="up",
            center_method="none",
        )

        # camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        # camera_to_world[..., 3] *= self.scale_factor
        # scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            # camera_to_worlds=camera_to_world,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        meta_data = {}
        # Load 3D points
        if self.config.load_3d_points:
            meta_data.update(self._load_3D_points())

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            features_filenames=features_filenames if len(features_filenames) > 0 else None,
            # alpha_color=alpha_color_tensor,
            # scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=meta_data,
        )

        return dataparser_outputs

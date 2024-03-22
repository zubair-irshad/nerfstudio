import json
import os

folder_path = '/home/zubairirshad/Downloads/front3d_single_scene/3dfront_0022_01/train'

path = os.path.join(folder_path, 'transforms.json')
# Read the original transforms.json file
with open(path, 'r') as file:
    data = json.load(file)

# Extract the first two items in the "frames" array
frames_val = data['frames'][:10]

# Create a new dictionary with the selected frames
new_data = {
    "camera_angle_x": data["camera_angle_x"],
    "camera_angle_y": data["camera_angle_y"],
    "fl_x": data["fl_x"],
    "fl_y": data["fl_y"],
    "k1": data["k1"],
    "k2": data["k2"],
    "p1": data["p1"],
    "p2": data["p2"],
    "cx": data["cx"],
    "cy": data["cy"],
    "w": data["w"],
    "h": data["h"],
    "aabb_scale": data["aabb_scale"],
    "scale": data["scale"],
    "offset": data["offset"],
    "room_bbox": data["room_bbox"],
    "num_room_objects": data["num_room_objects"],
    "frames": frames_val
}

# Write the new data to transforms_val.json
output_path = os.path.join(folder_path, 'transforms_val.json')
with open(output_path, 'w') as file:
    json.dump(new_data, file, indent=4)

print("transforms_val.json created successfully.")
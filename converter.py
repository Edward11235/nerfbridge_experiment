import torch
import numpy as np
from scipy.spatial.transform import Rotation
import nerfstudio.utils.poses as pose_utils

import argparse

import json
from pathlib import Path


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def pose_correction(matrix):
    static_transform = torch.zeros(3, 4)
    static_transform[:, :3] = torch.from_numpy(
        Rotation.from_euler("x", 180, degrees=True).as_matrix())

    pose_correction = torch.zeros(3, 4)
    pose_correction[:, :3] = torch.from_numpy(
        Rotation.from_euler("x", -90, degrees=True).as_matrix())

    matrix = torch.from_numpy(np.array(matrix).astype(np.float32))

    matrix[:3] = pose_utils.multiply(pose_correction, pose_utils.multiply((matrix[:3]), static_transform))
    return matrix


def matrices_to_camera_path(matrices: torch.Tensor, json_path: Path) -> None:
    assert matrices is not None
    assert matrices.size(1) == matrices.size(2) == 4
    matrices_flatten = matrices.reshape(matrices.size(0), -1).tolist()
    matrix_vertical_flatten = matrices[0].T.flatten().tolist()

    meta = {
        "keyframes": [{
            "matrix": str(matrix_vertical_flatten),
            "fov": 50,
            "aspect": 1,
            "properties": "[[\"FOV\",50],[\"NAME\",\"Camera 0\"],[\"TIME\",0]]"
        }],
        "camera_type": "perspective",
        "render_height": 480,
        "render_width": 640,
        "camera_path": [],
        "fps": 1,
        "seconds": matrices.size(0),
        "smoothness_value": 0.5,
        "is_cycle": False,
        "crop": None,
        }

    for idx in range(len(matrices_flatten)):
        meta["camera_path"].append({
            "camera_to_world": matrices_flatten[idx],
            "fov": 50,
            "aspect": 1
        })

    write_to_json(json_path, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a 4x4 matrix from a JSON file.")
    parser.add_argument("json_file", help="Path to a JSON file containing a 4x4 matrix.")
    args = parser.parse_args()

    try:
        meta = load_from_json(Path(args.json_file))
        matrix = meta["transform_matrix"]
        print("The input 4x4 matrix from JSON is:")
        print(matrix)
        print("The corrected 4x4 matrix from JSON is:")
        matrix = pose_correction(matrix)
        print(matrix.tolist())
        path = Path("camera_path.json")
        matrices_to_camera_path(matrix.unsqueeze(0), path)

        meta["transform_matrix"] = matrix.tolist()
        write_to_json(Path("matrix_converted.json"), meta)
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}")

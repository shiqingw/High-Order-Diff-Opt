import json
import sys
import os
from .quadrotor_2d import Quadrotor2D
from .quadrotor_3d import Quadrotor3D
from pathlib import Path

def get_system(system_name):
    with open(os.path.join(str(Path(__file__).parent), "system_params.json"), 'r') as f:
        data = json.load(f)
    if system_name not in data:
        raise ValueError("System name not found in system_params.json")
    data = data[system_name]
    if data["type"] == "Quadrotor2D":
        return Quadrotor2D(data["properties"], data["params"])
    elif data["type"] == "Quadrotor3D":
        return Quadrotor3D(data["properties"], data["params"])
    else:
        raise ValueError("System type not found in systems.json")



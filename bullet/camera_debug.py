import pybullet as p
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    "dist": 1.3,
    "yaw": 180,
    # "pitch": -41,
    "pitch": 0,
    "roll": 0,
    "look": [
        -0.35,
        -0.58,
        -0.88
    ],
    "fov": 60.0
})


def reset():

cid = p.connect(p.SHARED_MEMORY)
cid = p.connect(p.GUI)

p.resetDebugVisualizerCamera(cfg.dist, cfg.yaw, cfg.pitch, cfg.look)

import IPython; IPython.embed()


p.disconnect()
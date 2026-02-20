import mujoco
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco.viewer
import time
import mediapy as media


XML_PATH = "gemini_10modules.xml"

# ===== 载入模型 =====
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

viewer = mujoco.viewer.launch(model, data)

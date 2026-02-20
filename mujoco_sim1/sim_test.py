import numpy as np
import mujoco as mj
from mujoco import viewer
import os

MODEL_PATH = os.path.join("models", "E:/self_reproducing_robot/model_sim/single_module/urdf/single_module.urdf")

def sinus_target(t, amp_deg=30.0, freq_hz=0.8, bias_deg=0.0):
    return np.deg2rad(bias_deg + amp_deg * np.sin(2*np.pi*freq_hz*t))

def main():
    model = mj.MjModel.from_xml_path(MODEL_PATH)
    data = mj.MjData(model)

    # 提升接触稳定性（可调）
    model.opt.timestep = 0.002
    model.opt.integrator = mj.mjtIntegrator.mjINT_RK4

    with viewer.launch_passive(model, data) as v:
        t = 0.0
        while v.is_running():
            # 简单：把位置伺服的 ctrl 设置为目标角（弧度）
            q_target = sinus_target(data.time, amp_deg=30, freq_hz=1.0, bias_deg=0.0)
            # 只有一个 actuator：hip_pos
            data.ctrl[0] = q_target

            # 物理推进若干小步以匹配可视刷新
            for _ in range(5):
                mj.mj_step(model, data)

            # 在UI里显示一些调试量
            v.user_scn.ngeom = 0
            v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            v.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True

if __name__ == "__main__":
    main()

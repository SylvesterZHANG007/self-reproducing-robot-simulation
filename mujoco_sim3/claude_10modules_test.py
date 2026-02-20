"""
磁铁分裂/重连 无头控制脚本 (使用 eq_active0)
=============================================
model.eq_active0[id] → 默认状态, mj_resetData 时恢复
data.eq_active[id]   → 运行时状态, 动态开关

配合 10modules_magnetic.xml 使用
"""

import mujoco
import numpy as np

MAGNET_TORQUE_THRESHOLD = 0.15
REATTACH_DISTANCE = 0.025


def load(xml_path="claude_10modules.xml"):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 用 eq_active0 设置默认: 全部磁铁激活
    magnet_ids = []
    for i in range(1, 10):
        eq_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, f"magnet_{i}_{i+1}"
        )
        if eq_id >= 0:
            model.eq_active0[eq_id] = 1   # ★ 设置默认
            magnet_ids.append((i, i + 1, eq_id))

    # resetData 将 data.eq_active 从 model.eq_active0 复制
    mujoco.mj_resetData(model, data)

    return model, data, magnet_ids


def get_constraint_force(data, eq_id):
    total = 0.0
    for i in range(data.nefc):
        if data.efc_id[i] == eq_id:
            total += data.efc_force[i] ** 2
    return np.sqrt(total)


def check_separation(model, data, magnet_ids):
    for mod_a, mod_b, eq_id in magnet_ids:
        if not data.eq_active[eq_id]:       # ★ 读 data.eq_active
            continue
        force = get_constraint_force(data, eq_id)
        if force > MAGNET_TORQUE_THRESHOLD:
            data.eq_active[eq_id] = 0       # ★ 写 data.eq_active
            print(f"[SPLIT] {mod_a}<->{mod_b} (force={force:.4f})")


def check_reattach(model, data, magnet_ids, site_ids):
    for mod_a, mod_b, eq_id in magnet_ids:
        if data.eq_active[eq_id]:           # ★ 读 data.eq_active
            continue
        ek = f"{mod_a}_end"
        sk = f"{mod_b}_start"
        if ek not in site_ids or sk not in site_ids:
            continue
        d = np.linalg.norm(
            data.site_xpos[site_ids[ek]] - data.site_xpos[site_ids[sk]]
        )
        if d < REATTACH_DISTANCE:
            data.eq_active[eq_id] = 1       # ★ 写 data.eq_active
            print(f"[ATTACH] {mod_a}<->{mod_b} ({d*1e3:.1f}mm)")


if __name__ == "__main__":
    model, data, magnet_ids = load()

    # site IDs
    site_ids = {}
    for i in range(1, 11):
        for s in ["start", "end"]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"module{i}_{s}")
            if sid >= 0:
                site_ids[f"{i}_{s}"] = sid

    print(f"磁铁连接: {len(magnet_ids)}")
    print(f"eq_active0: {[model.eq_active0[eid] for _, _, eid in magnet_ids]}")
    print(f"eq_active:  {[data.eq_active[eid] for _, _, eid in magnet_ids]}")

    # 示例: 电机5全力扭
    m5 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_5")

    for step in range(10000):
        data.ctrl[m5] = 1.57
        mujoco.mj_step(model, data)

        if step % 100 == 0:
            check_separation(model, data, magnet_ids)
            check_reattach(model, data, magnet_ids, site_ids)

    # 最终状态
    print("\n最终连接:")
    for mod_a, mod_b, eq_id in magnet_ids:
        s = "●" if data.eq_active[eq_id] else "○"
        print(f"  {mod_a} {s} {mod_b}")

    # 演示: 重置恢复到 eq_active0
    print("\n重置...")
    mujoco.mj_resetData(model, data)
    print(f"eq_active (重置后): {[data.eq_active[eid] for _, _, eid in magnet_ids]}")
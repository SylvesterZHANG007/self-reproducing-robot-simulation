import mujoco
import mujoco.viewer
import numpy as np

# 加载上面的最新 XML 文件
xml_path = "gemini_10modules.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 动态监测配置
connections = []
for i in range(1, 10):
    weld_name = f"weld_{i}"
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    
    # 获取用于检测模块分离距离的 3 个方向的滑动关节 ID
    joint_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"sep_x_{i}")
    joint_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"sep_y_{i}")
    joint_z = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"sep_z_{i}")
    
    # 从 qposadr 中获取该关节在数据数组中的索引位置
    idx_x = model.jnt_qposadr[joint_x]
    idx_y = model.jnt_qposadr[joint_y]
    idx_z = model.jnt_qposadr[joint_z]
    
    connections.append({
        'weld_id': weld_id,
        'qpos_indices': [idx_x, idx_y, idx_z]
    })

# 物理断开阈值：2 毫米。这模拟了磁铁一旦被撬开 2mm，吸力就大幅衰减并断开
BREAK_THRESHOLD_METERS = 0.002 

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # 我们故意设置一个电机指令，让机器人自己在第 2 秒用力扭曲身体！
    motor_to_twist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'joint4_motor')
    
    while viewer.is_running():
        current_time = data.time
        
        # 1. 发送扭曲指令：2 秒后，电机 4 突然输出最大的扭矩
        if current_time > 2.0:
            data.ctrl[motor_to_twist] = 2.16  
            
        # 2. 实时监测所有磁铁结合点
        for conn in connections:
            weld_id = conn['weld_id']
            
            # 如果这个连接已经断开了，就不管它了
            if model.eq_active0[weld_id] == 0:
                continue
                
            # 读取当前这对外壳因为扭力拉扯而产生的形变/缝隙距离
            dx = data.qpos[conn['qpos_indices'][0]]
            dy = data.qpos[conn['qpos_indices'][1]]
            dz = data.qpos[conn['qpos_indices'][2]]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 3. 核心断开逻辑：如果外力撬开的缝隙 > 2mm
            if dist > BREAK_THRESHOLD_METERS:
                print(f"【磁力断开】连接点 {weld_id} 被外力撬开！缝隙达到了 {dist*1000:.2f} mm")
                model.eq_active0[weld_id] = 0 # 瞬间切断物理约束，完成分裂！
        
        mujoco.mj_step(model, data)
        viewer.sync()
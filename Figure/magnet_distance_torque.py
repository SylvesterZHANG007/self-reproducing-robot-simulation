import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# ========= PGF 配置：让 LaTeX 接管字体（IEEEtran -> Times New Roman） =========
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",    # 或 "xelatex"
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
})

# ========= 物理常量 & 参数 =========
mm_to_m   = 1e-3
inch_to_m = 0.0254
g   = 9.80665
mu0 = 4*math.pi*1e-7
Br  = 1.45                 # Tesla
L   = 106 * mm_to_m        # arm length
D   = 0.5 * inch_to_m      # magnet diameter = 0.5 in
R   = D/2
V   = (4/3)*math.pi*R**3
m_dipole = Br*V/mu0        # A·m^2

def F_of_r(r_m):           # N
    return (3*mu0*m_dipole**2)/(2*math.pi*r_m**4)

def tau_mag_Nm(r_m):       # N·m
    return 2 * F_of_r(r_m) * L

def Nm_to_kgcm(tau):       # kgf·cm
    return tau / 0.0980665

# ========= 曲线数据 =========
r_mm = np.linspace(16, 22, 50)
r_m  = r_mm * mm_to_m
tau_mag = Nm_to_kgcm(np.array([tau_mag_Nm(r) for r in r_m]))
tau_g_1kg = Nm_to_kgcm(1.0 * g * L)
tau_0kg = tau_mag
tau_1kg = tau_mag + tau_g_1kg

# ========= 标注点（红色 ×） =========
r_mark = 19.5                       # mm
tau_mark_0 = Nm_to_kgcm(tau_mag_Nm(r_mark*mm_to_m))         # ≈ 13.739 kg·cm
tau_mark_1 = tau_mark_0 + tau_g_1kg                          # ≈ 24.339 kg·cm

# ========= 画图 =========
plt.figure(figsize=(4.2,3.2))
plt.plot(r_mm, tau_0kg, label="Arm mass 0 kg")
plt.plot(r_mm, tau_1kg, label="Arm mass 1 kg")

# 红色×与数值标注（做一点点偏移，避免遮挡曲线）
plt.scatter([r_mark, r_mark], [tau_mark_0, tau_mark_1], marker="x", s=55, c="red")
plt.annotate(fr"{tau_mark_0:.1f} kg$\cdot$cm",
             (r_mark, tau_mark_0), xytext=(+4, +6),
             textcoords="offset points", ha="left", va="top", color="red")
plt.annotate(fr"{tau_mark_1:.1f} kg$\cdot$cm",
             (r_mark, tau_mark_1), xytext=(+0.6, +6),
             textcoords="offset points", ha="left", va="bottom", color="red")

plt.xlabel(r"Magnet center distance $r$ (mm)")
plt.ylabel(r"Required servo torque (kg$\cdot$cm)")
# plt.title("Required Servo Torque vs. Magnet Distance")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 关键数值打印到终端
print(f"r = {r_mark} mm")
print(f"tau_mag = {tau_mark_0:.4f} kg·cm")
print(f"tau_total (1 kg) = {tau_mark_1:.4f} kg·cm (adds {tau_g_1kg:.4f} kg·cm)")

# ========= 保存 PGF（裁掉多余白边，防止 LaTeX 中看起来不居中） =========
plt.savefig("torque_vs_r.pgf", bbox_inches="tight", pad_inches=0)
print("Saved: torque_vs_r.pgf")

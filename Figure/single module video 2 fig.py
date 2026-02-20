import cv2
import numpy as np
import argparse
import os

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def lighten_blend(base, frame):
    # 像“Lighten”混合：逐通道取最大值
    return np.maximum(base, frame)

def alpha_blend(base, frame, alpha):
    # 带透明度的线性叠加
    return cv2.addWeighted(frame, alpha, base, 1.0 - alpha, 0)

def foreground_mask_mog2(frame, bgsubtractor, ksize_open=3, ksize_close=5):
    fgmask = bgsubtractor.apply(frame)
    # 去噪：开运算去小噪点，闭运算填小孔
    if ksize_open > 0:
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((ksize_open, ksize_open), np.uint8))
    if ksize_close > 0:
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((ksize_close, ksize_close), np.uint8))
    # 二值化
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    return fgmask

def color_map_trail(index, total, cmap=cv2.COLORMAP_JET):
    # 生成时间色带颜色（早→晚）
    if total <= 1:
        v = 255
    else:
        v = int(255.0 * index / (total - 1))
    color = cv2.applyColorMap(np.array([[v]], dtype=np.uint8), cmap)[0,0]  # BGR
    return (int(color[0]), int(color[1]), int(color[2]))

def main():
    parser = argparse.ArgumentParser(description="Merge all motion frames into a single long-exposure image.")
    parser.add_argument("--video", required=True, help="Path to the input video file (e.g., robot.mp4)")
    parser.add_argument("--mode", default="lighten", choices=["lighten", "alpha", "fgmask"],
                        help="Blend mode: 'lighten' (recommended), 'alpha', or 'fgmask' (foreground only).")
    parser.add_argument("--step", type=int, default=1, help="Use every N-th frame to speed up and reduce ghosting.")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor for processing (e.g., 0.5).")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive); -1 = till end.")
    parser.add_argument("--alpha_min", type=float, default=0.05, help="Min alpha for earliest frame (alpha mode).")
    parser.add_argument("--alpha_max", type=float, default=0.9, help="Max alpha for latest frame (alpha mode).")
    parser.add_argument("--out", default="output_long_exposure.png", help="Path to save the final composited image.")
    parser.add_argument("--out_trail", default="output_trail_colormap.png", help="Path to save a time-colored trail image.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, args.start)
    end = total_frames if args.end < 0 else min(args.end, total_frames)
    indices = list(range(start, end, args.step))
    if len(indices) == 0:
        raise ValueError("No frames selected. Check --start/--end/--step parameters.")

    # 读第一帧，确定尺寸
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Failed to read the first selected frame.")
    if args.resize != 1.0:
        first = cv2.resize(first, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

    h, w = first.shape[:2]
    base = np.zeros_like(first)          # 合成底图（黑底）
    trail = np.zeros_like(first)         # 时间色带轨迹（可选输出）
    trail_mask_accum = np.zeros((h, w), dtype=np.uint8)  # 只做展示，不用于逻辑

    # 背景建模器（仅 fgmask 模式用）
    bgsub = None
    if args.mode == "fgmask":
        bgsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

    # 主循环
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if args.resize != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        if args.mode == "lighten":
            base = lighten_blend(base, frame)

        elif args.mode == "alpha":
            # 线性时间权重：早帧低 alpha，晚帧高 alpha
            if len(indices) == 1:
                alpha = args.alpha_max
            else:
                alpha = args.alpha_min + (args.alpha_max - args.alpha_min) * (i / (len(indices) - 1))
            base = alpha_blend(base, frame, alpha)

        elif args.mode == "fgmask":
            fgmask = foreground_mask_mog2(frame, bgsub)
            # 只把前景区域复制到 base（保持“最后更亮”的效果）
            fg = cv2.bitwise_and(frame, frame, mask=fgmask)
            # 用最大值混合减少覆盖丢失细节
            base = np.maximum(base, fg)

        # —— 时间色带轨迹图（可选增强）——
        # 用边缘或前景做细轨迹，减少大片噪声
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        # 也可以用 fgmask（如果模式是 fgmask），否则用 edges
        mask_for_trail = edges if args.mode != "fgmask" else foreground_mask_mog2(frame, bgsub, 0, 0)
        color = color_map_trail(i, len(indices))  # BGR
        colored = np.zeros_like(frame)
        colored[:] = color
        colored = cv2.bitwise_and(colored, colored, mask=mask_for_trail)
        trail = np.maximum(trail, colored)
        # 为了更清晰，可以稍微膨胀一下轨迹
        trail_mask_accum = np.maximum(trail_mask_accum, mask_for_trail)

    cap.release()

    # 保存输出
    ensure_dir(args.out)
    ensure_dir(args.out_trail)
    cv2.imwrite(args.out, base)
    # 时间色带图和原合成叠加，提升可读性（可注释掉）
    trail_overlay = cv2.addWeighted(base, 0.7, trail, 0.9, 0)
    cv2.imwrite(args.out_trail, trail_overlay)

    print(f"Saved: {args.out}")
    print(f"Saved: {args.out_trail}")
    print("Done.")

if __name__ == "__main__":
    main()

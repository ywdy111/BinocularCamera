import numpy as np
import cv2
import os

# ==========================================
# 1. 双频四步相移条纹图
# ==========================================
def generate_dual_frequency_patterns(width=1280, height=720, p1=80, p2=40, out_dir="config\\data_stripe\\dual_freq_patterns"):
    """
    生成双频四步相移条纹图 (2 frequencies * 4 steps = 8 frames)
    """
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(width)
    print(f"开始生成双频四步条纹图，分辨率: {width}x{height}")
    print(f"频率1 (周期 P={p1}), 频率2 (周期 P={p2})")
    
    phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]
    
    # 频率 1
    for i, shift in enumerate(phase_shifts):
        intensity = 127.5 + 127.5 * np.cos(2 * np.pi * x / p1 + shift)
        pattern_2d = np.tile(intensity, (height, 1)).astype(np.uint8)
        filename = os.path.join(out_dir, f"01_freq1_P{p1}_step{i+1}.bmp")
        cv2.imwrite(filename, pattern_2d)
        print(f"  已生成低频: {filename}")

    # 频率 2
    for i, shift in enumerate(phase_shifts):
        intensity = 127.5 + 127.5 * np.cos(2 * np.pi * x / p2 + shift)
        pattern_2d = np.tile(intensity, (height, 1)).astype(np.uint8)
        filename = os.path.join(out_dir, f"02_freq2_P{p2}_step{i+1}.bmp")
        cv2.imwrite(filename, pattern_2d)
        print(f"  已生成高频: {filename}")
        
    print(f"-> 全部 8 张图像已保存至 '{out_dir}'\n")


# ==========================================
# 2. 互补格雷码 + 四步相移 (9张图)
# ==========================================
def generate_cgc_patterns(width=1280, height=720, periods=16, out_dir="config\\data_stripe\\patterns_9_frames_cgc"):
    """
    生成 9 张互补格雷码高精度条纹图
    包含: 4步相移 + 4级基础格雷码 + 1张互补格雷码
    """
    os.makedirs(out_dir, exist_ok=True)
    P = width / periods
    x = np.arange(width)
    print(f"开始生成互补格雷码条纹图，分辨率: {width}x{height}，周期数: {periods}")
    
    # 1. 4 步相移
    for i in range(4):
        phase_shift = i * (np.pi / 2)
        intensity = 127.5 + 127.5 * np.cos(2 * np.pi * x / P + phase_shift)
        pattern_2d = np.tile(intensity, (height, 1)).astype(np.uint8)
        filename = os.path.join(out_dir, f"0{i+1}_phase_shift_step{i+1}.bmp")
        cv2.imwrite(filename, pattern_2d)
        print(f"  已生成相移图: {filename}")

    # 2. 4 张基础格雷码
    region_idx = (x / P).astype(int)
    gray_code = region_idx ^ (region_idx >> 1)
    
    for i in range(4):
        bit_pos = 3 - i 
        bit_val = ((gray_code >> bit_pos) & 1) * 255
        pattern_2d = np.tile(bit_val, (height, 1)).astype(np.uint8)
        filename = os.path.join(out_dir, f"0{i+5}_gray_code_bit{bit_pos}.bmp")
        cv2.imwrite(filename, pattern_2d)
        print(f"  已生成格雷码: {filename}")

    # 3. 1 张互补格雷码
    region_idx_comp = ((x + P / 2) / P).astype(int)
    gray_code_comp = region_idx_comp ^ (region_idx_comp >> 1)
    comp_val = ((gray_code_comp >> 0) & 1) * 255
    pattern_2d = np.tile(comp_val, (height, 1)).astype(np.uint8)
    filename = os.path.join(out_dir, f"09_complementary_gray.bmp")
    cv2.imwrite(filename, pattern_2d)
    print(f"  已生成互补码: {filename}")
    
    print(f"-> 全部 9 张图像已保存至 '{out_dir}'\n")


# ==========================================
# 通用多频多步生成器 (内部调用)
# ==========================================
def _generate_multi_freq_steps(width, height, periods, steps, out_dir):
    """
    通用函数：用于生成任意频数、任意步数的相移条纹
    """
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(width)
    phase_shifts = [i * (2 * np.pi / steps) for i in range(steps)]
    
    print(f"开始生成 {len(periods)}频 {steps}步 条纹图，分辨率: {width}x{height}")
    print(f"总计 {len(periods) * steps} 张图案...")
    
    frame_index = 1
    for freq_idx, P in enumerate(periods):
        print(f"---> 频率组 {freq_idx + 1} (周期 P = {P})")
        for step_idx, shift in enumerate(phase_shifts):
            intensity = 127.5 + 127.5 * np.cos(2 * np.pi * x / P + shift)
            pattern_2d = np.tile(intensity, (height, 1)).astype(np.uint8)
            filename = os.path.join(out_dir, f"{frame_index:02d}_Freq{freq_idx+1}_P{P}_step{step_idx+1:02d}.bmp")
            cv2.imwrite(filename, pattern_2d)
            print(f"    已生成: {filename}")
            frame_index += 1
            
    print(f"-> 全部 {frame_index - 1} 张图像已保存至 '{out_dir}'\n")


# ==========================================
# 3. 三频十二步相移
# ==========================================
def generate_3freq_12step_patterns(width=1280, height=720, periods=None, out_dir="config\\data_stripe\\3freq_12step_patterns"):
    if periods is None: periods = [70, 75, 80]
    _generate_multi_freq_steps(width, height, periods, 12, out_dir)

# ==========================================
# 4. 三频六步相移
# ==========================================
def generate_3freq_6step_patterns(width=1280, height=720, periods=None, out_dir="config\\data_stripe\\3freq_6step_patterns"):
    if periods is None: periods = [70, 75, 80]
    _generate_multi_freq_steps(width, height, periods, 6, out_dir)

# ==========================================
# 5. 三频四步相移
# ==========================================
def generate_3freq_4step_patterns(width=1280, height=720, periods=None, out_dir="config\\data_stripe\\3freq_4step_patterns"):
    if periods is None: periods = [70, 75, 80]
    _generate_multi_freq_steps(width, height, periods, 4, out_dir)

# ==========================================
# 6. 三频三步相移
# ==========================================
def generate_3freq_3step_patterns(width=1280, height=720, periods=None, out_dir="config\\data_stripe\\3freq_3step_patterns"):
    if periods is None: periods = [70, 75, 80]  # 你原代码里设定的较小周期
    _generate_multi_freq_steps(width, height, periods, 3, out_dir)


if __name__ == "__main__":
    # 分辨率设置
    W, H = 1280, 720
    
    # 你可以取消注释来生成对应的条纹图集：
    
    # 1. 生成双频四步
    generate_dual_frequency_patterns(width=W, height=H, p1=80, p2=40)
    
    # 2. 生成互补格雷码 (9张图)
    generate_cgc_patterns(width=W, height=H, periods=16)
    
    # 3. 生成三频十二步 (36张图)
    generate_3freq_12step_patterns(width=W, height=H, periods=[70, 75, 80])
    
    # 4. 生成三频六步 (18张图)
    generate_3freq_6step_patterns(width=W, height=H, periods=[70, 75, 80])
    
    # 5. 生成三频四步 (12张图)
    generate_3freq_4step_patterns(width=W, height=H, periods=[70, 75, 80])
    
    # 6. 生成三频三步 (9张图)
    generate_3freq_3step_patterns(width=W, height=H, periods=[70, 75, 80])
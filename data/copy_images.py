import os
import shutil
import glob

def collect_and_rename_images():
    # 定义基础路径和目标路径
    base_dir = r"C:\Users\ywdy\Desktop\BinocularCamera\data"
    dataset_dir = os.path.join(base_dir, "dataset")

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"已创建目标文件夹: {dataset_dir}")

    # 初始化重命名计数器
    counter = 1

    # 定义需要匹配的文件名模式
    file_patterns = [
        "wrapped_phase_f80_step4_binary_*.png",
        "wrapped_phase_f160_step4_binary_*.png"
    ]

    # 遍历 1 到 66 号文件夹
    for i in range(1, 67):
        folder_num = str(i)
        
        # 遍历 left 和 right 两个子目录
        for side in ["left", "right"]:
            # 构建目标搜索路径: data\1\rebuild\left 等
            search_path = os.path.join(base_dir, folder_num, "rebuild", side)
            
            # 如果该路径不存在，则跳过
            if not os.path.exists(search_path):
                continue
                
            # 在当前路径下匹配两种前缀的图片
            for pattern in file_patterns:
                # 使用 glob 获取所有匹配的完整文件路径
                full_pattern = os.path.join(search_path, pattern)
                matched_files = glob.glob(full_pattern)
                
                for file_path in matched_files:
                    # 构建新的文件名（纯阿拉伯数字）和完整目标路径
                    new_filename = f"{counter}.png"
                    dest_path = os.path.join(dataset_dir, new_filename)
                    
                    # 执行复制操作
                    shutil.copy2(file_path, dest_path)
                    print(f"正在复制: {file_path}  ->  {new_filename}")
                    
                    # 计数器加 1，准备命名下一个文件
                    counter += 1

    print("-" * 50)
    print(f"复制与重命名完成！共成功处理了 {counter - 1} 个文件。")
    print(f"所有文件已保存至: {dataset_dir}")

import cv2
import numpy as np

def convert_binary_dataset_for_sam():
    # 原始二值化数据集路径
    input_dir = r"C:\Users\ywdy\Desktop\BinocularCamera\data\dataset"
    # 转换后供 SAM 使用的新文件夹
    output_dir = r"C:\Users\ywdy\Desktop\BinocularCamera\data\dataset_sam_ready"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # 获取所有 png 文件
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_paths:
        print("未在目标路径下找到 PNG 图片，请检查路径是否正确。")
        return

    success_count = 0

    for path in image_paths:
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        
        # 1. 强制以灰度（单通道）模式读取二值化图像
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"无法读取图片: {filename}")
            continue
            
        # 2. 规范化像素值：确保其为标准的 uint8 类型，且只有 0 和 255
        # 应对部分特殊保存格式（如存成了 0 和 1 的浮点数或 bool 值）
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # 3. 将单通道 (H, W) 复制成三通道 (H, W, 3) 
        # 虽然视觉上它依然是黑白二值图，但维度上已经满足了 SAM 的输入标准
        img_3c = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 4. 保存为标准图像
        # 因为 R、G、B 三个通道的值完全一致（同为 0 或同为 255），所以直接用 cv2 保存即可
        cv2.imwrite(out_path, img_3c)
        success_count += 1
        print(f"已成功转换二值图: {filename}")

    print("-" * 50)
    print(f"数据转换完成！成功处理了 {success_count} 个二值化文件。")
    print(f"请在 SAM 推理代码中读取此路径下的数据：\n--> {output_dir}")

if __name__ == "__main__":
    # convert_binary_dataset_for_sam()
    collect_and_rename_images()
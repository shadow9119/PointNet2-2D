import os
import numpy as np
import glob
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
# 输入文件夹路径
INPUT_DIR = 'data/real_day'
# 输出文件夹路径
OUTPUT_DIR = 'data/real_day_split'
# 每个切片包含的点数 (与你的模型 npoint 保持一致)
CHUNK_SIZE = 16384
# 丢弃阈值：如果最后一个切片点数少于此值，则丢弃（避免太小的碎片）
# 建议设为 CHUNK_SIZE 的 10% 或 20%，或者设为 0 保留所有
MIN_POINTS_THRESHOLD = 1000 
# ===========================================

def split_files():
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[Info] 创建输出目录: {OUTPUT_DIR}")
    else:
        print(f"[Info] 输出目录已存在: {OUTPUT_DIR} (新文件将写入此处)")

    # 2. 获取所有 txt 文件
    file_list = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    if len(file_list) == 0:
        print(f"[Error] 在 {INPUT_DIR} 下没有找到 .txt 文件！")
        return

    print(f"[Info] 找到 {len(file_list)} 个原始文件，开始切片处理...")
    
    total_chunks = 0
    total_original_points = 0

    # 3. 遍历处理
    for file_path in tqdm(file_list, desc="Processing"):
        try:
            # 读取数据 (使用 float64 保持精度)
            # 格式: [Y(沿轨), Z(高程), ID, Label]
            data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
            
            # 记录原始点数
            num_points = data.shape[0]
            total_original_points += num_points
            
            # 【关键步骤】按第一列 (Y轴/沿轨方向) 进行排序
            # 保证切片后的数据在空间上是连续的，保留地形特征
            sorted_indices = np.argsort(data[:, 0])
            data = data[sorted_indices]

            # 计算切片数量
            # 使用 ceil 向上取整，或者直接整除看剩余
            # 这里采用简单的滑动窗口逻辑
            num_chunks = int(np.ceil(num_points / CHUNK_SIZE))
            
            base_filename = os.path.splitext(os.path.basename(file_path))[0]

            for i in range(num_chunks):
                start_idx = i * CHUNK_SIZE
                end_idx = min((i + 1) * CHUNK_SIZE, num_points)
                
                chunk_data = data[start_idx:end_idx, :]
                
                # 检查点数是否过少
                if chunk_data.shape[0] < MIN_POINTS_THRESHOLD:
                    continue

                # 构造新文件名: 原文件名_part_00x.txt
                save_name = f"{base_filename}_part_{i:03d}.txt"
                save_path = os.path.join(OUTPUT_DIR, save_name)

                # 保存文件
                # fmt='%.6f' 保证坐标精度，最后一列标签其实是整数，但用 float 保存也没问题
                # data[:, 2] (ID) 和 data[:, 3] (Label) 也会被保存
                np.savetxt(save_path, chunk_data, delimiter=',', fmt='%.6f')
                
                total_chunks += 1

        except Exception as e:
            print(f"\n[Error] 处理文件 {file_path} 时出错: {e}")

    print("\n" + "="*50)
    print(f"处理完成！")
    print(f"原始文件数: {len(file_list)}")
    print(f"原始总点数: {total_original_points}")
    print(f"生成切片数: {total_chunks}")
    print(f"切片大小: {CHUNK_SIZE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*50)

if __name__ == '__main__':
    split_files()
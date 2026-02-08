#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：统计指定路径下所有txt文件中第四列的0和1的个数及占比
"""

import os
import glob

def count_ones_zeros_in_file(file_path):
    """
    统计单个txt文件中第四列的0和1的个数及占比
    第四列: 0=噪声, 1=信号
    """
    noise_count = 0  # 噪声（0）
    signal_count = 0  # 信号（1）
    total_points = 0
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        fourth_col = int(float(parts[3]))  # 第四列
                        if fourth_col == 0:
                            noise_count += 1
                        elif fourth_col == 1:
                            signal_count += 1
                        total_points += 1
                    except (ValueError, IndexError):
                        continue
                        
        if total_points > 0:
            noise_percentage = (noise_count / total_points) * 100
            signal_percentage = (signal_count / total_points) * 100
        else:
            noise_percentage = signal_percentage = 0
            
        return noise_count, signal_count, total_points, noise_percentage, signal_percentage
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return 0, 0, 0, 0, 0

def process_all_files(root_dir):
    """
    处理指定目录下的所有txt文件
    """
    # 获取所有txt文件的路径
    file_pattern = os.path.join(root_dir, '**/*.txt')
    txt_files = glob.glob(file_pattern, recursive=True)
    
    if not txt_files:
        print(f"在 {root_dir} 目录及其子目录中未找到任何txt文件")
        return
    
    # 创建报告内容
    report_lines = []
    report_lines.append("=" * 150)
    report_lines.append("点云数据统计报告 - 噪声与信号分析")
    report_lines.append(f"根目录: {root_dir}")
    report_lines.append(f"处理文件数: {len(txt_files)}")
    report_lines.append(f"生成时间: {os.popen('date').read().strip()}")
    report_lines.append("=" * 150)
    report_lines.append("")
    report_lines.append("说明: 第四列标签 - 0=噪声点, 1=信号点")
    report_lines.append("")
    
    # 表头
    header = f"{'文件路径':<80} | {'总点数':<10} | {'噪声点数':<10} | {'信号点数':<10} | {'噪声占比(%)':<12} | {'信号占比(%)':<12}"
    report_lines.append(header)
    report_lines.append("-" * 150)
    
    # 统计总和
    total_noise = 0
    total_signal = 0
    total_points_all = 0
    
    # 处理每个文件
    for file_path in sorted(txt_files):
        noise_count, signal_count, total_points, noise_percentage, signal_percentage = count_ones_zeros_in_file(file_path)
        
        # 只显示文件名（去掉根目录部分）
        relative_path = os.path.relpath(file_path, root_dir)
        
        report_lines.append(
            f"{relative_path:<80} | {total_points:<10} | {noise_count:<10} | {signal_count:<10} | "
            f"{noise_percentage:<12.2f} | {signal_percentage:<12.2f}"
        )
        
        total_noise += noise_count
        total_signal += signal_count
        total_points_all += total_points
    
    # 添加汇总信息
    report_lines.append("-" * 150)
    if total_points_all > 0:
        total_noise_percentage = (total_noise / total_points_all) * 100
        total_signal_percentage = (total_signal / total_points_all) * 100
    else:
        total_noise_percentage = total_signal_percentage = 0
        
    report_lines.append(
        f"{'总计':<80} | {total_points_all:<10} | {total_noise:<10} | {total_signal:<10} | "
        f"{total_noise_percentage:<12.2f} | {total_signal_percentage:<12.2f}"
    )
    report_lines.append("")
    report_lines.append("=" * 150)
    report_lines.append("")
    report_lines.append("汇总统计:")
    report_lines.append(f"  - 总文件数: {len(txt_files)}")
    report_lines.append(f"  - 总点数: {total_points_all:,}")
    report_lines.append(f"  - 噪声点总数: {total_noise:,} (占比: {total_noise_percentage:.2f}%)")
    report_lines.append(f"  - 信号点总数: {total_signal:,} (占比: {total_signal_percentage:.2f}%)")
    report_lines.append(f"  - 平均每个文件点数: {total_points_all / len(txt_files):.0f}")
    if total_signal > 0:
        report_lines.append(f"  - 噪声/信号比: {total_noise / total_signal:.4f}")
    report_lines.append("")
    report_lines.append("=" * 150)
    
    # 写入报告文件
    report_path = os.path.join(root_dir, '数据统计报告.txt')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('\n'.join(report_lines))
    
    # 控制台输出
    print("\n" + "=" * 80)
    print("统计完成！")
    print("=" * 80)
    print(f"报告已生成: {report_path}")
    print(f"\n汇总信息:")
    print(f"  - 共处理文件数: {len(txt_files)}")
    print(f"  - 总点数: {total_points_all:,}")
    print(f"  - 噪声点: {total_noise:,} (占比: {total_noise_percentage:.2f}%)")
    print(f"  - 信号点: {total_signal:,} (占比: {total_signal_percentage:.2f}%)")
    print(f"  - 平均每个文件点数: {total_points_all / len(txt_files):.0f}")
    if total_signal > 0:
        print(f"  - 噪声/信号比: {total_noise / total_signal:.4f}")
    print("=" * 80 + "\n")
    
    return report_path

if __name__ == "__main__":
    # 设置要处理的根目录
    root_directory = r"/data/home/stoniachen/code/PointNet2-2D/data/real_water"
    
    # 处理所有文件并生成报告
    report_file = process_all_files(root_directory)
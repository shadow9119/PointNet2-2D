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
    """
    zeros = 0
    ones = 0
    total_lines = 0
    
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
                            zeros += 1
                        elif fourth_col == 1:
                            ones += 1
                        total_lines += 1
                    except (ValueError, IndexError):
                        continue
                        
        if total_lines > 0:
            zero_percentage = (zeros / total_lines) * 100
            one_percentage = (ones / total_lines) * 100
        else:
            zero_percentage = one_percentage = 0
            
        return zeros, ones, total_lines, zero_percentage, one_percentage
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
    report_lines.append("=" * 80)
    report_lines.append("数据统计报告")
    report_lines.append(f"根目录: {root_dir}")
    report_lines.append(f"处理文件数: {len(txt_files)}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 表头
    report_lines.append(f"{'文件路径':<80} | {'0的个数':<10} | {'1的个数':<10} | {'总行数':<10} | {'0占比(%)':<10} | {'1占比(%)':<10}")
    report_lines.append("-" * 140)
    
    # 统计总和
    total_zeros = 0
    total_ones = 0
    total_lines_all = 0
    
    # 处理每个文件
    for file_path in sorted(txt_files):
        zeros, ones, total_lines, zero_percentage, one_percentage = count_ones_zeros_in_file(file_path)
        
        # 只显示文件名（去掉根目录部分）
        relative_path = os.path.relpath(file_path, root_dir)
        
        report_lines.append(f"{relative_path:<80} | {zeros:<10} | {ones:<10} | {total_lines:<10} | {zero_percentage:<10.2f} | {one_percentage:<10.2f}")
        
        total_zeros += zeros
        total_ones += ones
        total_lines_all += total_lines
    
    # 添加汇总信息
    report_lines.append("-" * 140)
    if total_lines_all > 0:
        total_zero_percentage = (total_zeros / total_lines_all) * 100
        total_one_percentage = (total_ones / total_lines_all) * 100
    else:
        total_zero_percentage = total_one_percentage = 0
        
    report_lines.append(f"{'总计':<80} | {total_zeros:<10} | {total_ones:<10} | {total_lines_all:<10} | {total_zero_percentage:<10.2f} | {total_one_percentage:<10.2f}")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 写入报告文件
    report_path = os.path.join(root_dir, '数据统计报告.txt')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('\n'.join(report_lines))
    
    print(f"报告已生成: {report_path}")
    print(f"共处理 {len(txt_files)} 个文件")
    print(f"总计: 0的个数: {total_zeros}, 1的个数: {total_ones}, 总行数: {total_lines_all}")
    print(f"总计占比: 0占 {total_zero_percentage:.2f}%, 1占 {total_one_percentage:.2f}%")
    
    return report_path

if __name__ == "__main__":
    # 设置要处理的根目录
    root_directory = r"C:\Users\14711\Desktop\PointNet2-main\data\simulated_code"
    
    # 处理所有文件并生成报告
    report_file = process_all_files(root_directory)
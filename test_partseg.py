import argparse
import os
from pathlib import Path
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import gc  # 添加垃圾回收模块

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'signal': [0, 1]}
seg_label_to_cat = {label: cat for cat in seg_classes for label in seg_classes[cat]}

def plot_points(global_index, points, pred_choice, dataset_name, filename, precision=None, recall=None, f1=None, exp_name=None):
    pred_choice = np.array(pred_choice, dtype=int)
    color_map = {0: [0.93, 0.69, 0.13], 1: [0.49, 0.18, 0.56]}
    colors = np.array([color_map[label] for label in pred_choice])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, s=1, alpha=0.7)
    
    # 在标题中包含R、P、F值
    if precision is not None and recall is not None and f1 is not None:
        title = f'{dataset_name.capitalize()} Set - {filename}\nP: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}'
    else:
        title = f'{dataset_name.capitalize()} Set - {filename}'
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Noise', markersize=5, markerfacecolor=[0.93, 0.69, 0.13]),
        plt.Line2D([0], [0], marker='o', color='w', label='Signal', markersize=5, markerfacecolor=[0.49, 0.18, 0.56])
    ], title="Classes")

    # 修改保存路径，使用基于实验名称的目录结构
    if exp_name:
        save_path = os.path.join(BASE_DIR, 'results', exp_name, 'test/')
    else:
        save_path = os.path.join(BASE_DIR, 'results/test/')
    os.makedirs(save_path, exist_ok=True)
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(save_path, f'{base_filename}.svg')
    
    plt.savefig(output_path, bbox_inches='tight')
    
    # 显式清理图形资源
    plt.close(fig)
    del fig, ax, scatter
    gc.collect()

def to_categorical(y, num_classes):
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    return new_y.to(y.device)

def pc_denormalize(pc, pc_min, pc_max):
    for i in range(pc.shape[1]):
        pc[:, i] = (pc[:, i] + 1) / 2 * (pc_max[i] - pc_min[i]) + pc_min[i]
    return pc

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=None, help='point Number, set None for all points')
    parser.add_argument('--max_point', type=int, default=50000, help='maximum point number to avoid OOM, points will be randomly sampled if exceeded')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--ckpt', type=str, default=None, help='model checkpoint')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--data_root', type=str, required=True, help='data root file')
    parser.add_argument('--output', action='store_false', help='output test results')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold')
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name for CSV output')
    parser.add_argument('--conf_channel', action='store_true', default=False, help='use confidence channel (x, y, signal_conf) instead of just (x, y)')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # 设置CUDA设备并清理缓存
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = os.path.join('log', 'part_seg', args.log_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'eval.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    if args.output:
        # 修改输出目录创建逻辑，使用基于实验名称的目录结构
        if args.exp_name:
            if args.ckpt:
                output_dir = Path(os.path.join('results', args.exp_name, 'output_' + str(args.ckpt).split('.')[0] + '_' + str(args.threshold)))
            else:
                output_dir = Path(os.path.join('results', args.exp_name, 'output_' + str(args.threshold)))
        else:
            if args.ckpt:
                output_dir = Path(experiment_dir + '/output_' + str(args.ckpt).split('.')[0] + '_' + str(args.threshold))
            else:
                output_dir = Path(experiment_dir + '/output_' + str(args.threshold))

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    root = args.data_root

    # 减少num_workers以降低内存使用
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', conf_channel=args.conf_channel)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=2)
    log_string("test_partseg中，已读入test集个数: %d" % len(TEST_DATASET))
    
    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    model_name = os.listdir(os.path.join(experiment_dir, 'logs'))[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, conf_channel=args.conf_channel).to(device)
    
    # 加载检查点时清理缓存
    if args.ckpt:
        checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints', args.ckpt))
    else:
        checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints/model.pth'))
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint  # 删除不再需要的检查点
    torch.cuda.empty_cache()

    thres = args.threshold

    with torch.no_grad():
        tp_acc, fp_acc, fn_acc = 0, 0, 0
        test_metrics = {}
        classifier = classifier.eval()
        global_index = 0
        results = []

        for batch_id, (points, label, target, point_set_normalized_mask, pc_min, pc_max, fn) in enumerate(testDataLoader):
            cur_batch_size, NUM_POINT, _ = points.size()
            
            # 处理超大点云：如果点云数量超过max_point，进行随机采样
            if args.max_point is not None and NUM_POINT > args.max_point:
                # 在命令行和日志中输出警告信息
                warning_msg = f'\n{"="*80}\n⚠️  WARNING: Large Point Cloud Detected!\n{"-"*80}'
                warning_msg += f'\nFile: {fn[0]}'
                warning_msg += f'\nOriginal Points: {NUM_POINT:,}'
                warning_msg += f'\nMax Allowed: {args.max_point:,}'
                warning_msg += f'\nAction: Randomly sampling {args.max_point:,} points from {NUM_POINT:,} points'
                warning_msg += f'\n{"-"*80}\n'
                print(warning_msg)
                log_string(warning_msg)
                # 对batch中的每个样本进行采样
                sampled_points_list = []
                sampled_target_list = []
                sampled_mask_list = []
                
                for i in range(cur_batch_size):
                    # 获取当前样本的有效点（根据mask）
                    cur_mask = point_set_normalized_mask[i].numpy()
                    valid_indices = np.where(cur_mask)[0]
                    
                    if len(valid_indices) > args.max_point:
                        # 随机采样
                        sampled_indices = np.random.choice(valid_indices, args.max_point, replace=False)
                        sampled_indices = np.sort(sampled_indices)  # 保持顺序
                        
                        # 创建新的点云数据
                        new_points = points[i, sampled_indices, :].unsqueeze(0)
                        new_target = target[i, sampled_indices].unsqueeze(0)
                        new_mask = torch.ones(args.max_point, dtype=torch.bool)
                    else:
                        # 不需要采样
                        new_points = points[i, :len(valid_indices), :].unsqueeze(0)
                        new_target = target[i, :len(valid_indices)].unsqueeze(0)
                        new_mask = torch.ones(len(valid_indices), dtype=torch.bool)
                    
                    sampled_points_list.append(new_points)
                    sampled_target_list.append(new_target)
                    sampled_mask_list.append(new_mask)
                
                # 重新组合batch
                points = torch.cat(sampled_points_list, dim=0)
                target = torch.cat(sampled_target_list, dim=0)
                point_set_normalized_mask = torch.stack(sampled_mask_list, dim=0)
                NUM_POINT = points.size(1)
            
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            
            # 初始化投票池并立即使用
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).to(device)
            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred
                del seg_pred  # 及时删除中间变量
            
            seg_pred = vote_pool / args.num_votes
            cur_pred = seg_pred.cpu().numpy()
            del vote_pool, seg_pred
            
            # 预分配数组
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT), dtype=np.int32)
            cur_pred_prob = np.zeros((cur_batch_size, NUM_POINT), dtype=np.float64)
            
            target_np = target.cpu().data.numpy()
            point_set_normalized_mask_np = point_set_normalized_mask.numpy()
            
            cur_pred_prob_mask = []
            cur_pred_val_mask = []
            target_mask = []

            for i in range(cur_batch_size):
                prob = np.exp(cur_pred[i, :, :])
                cur_pred_prob[i, :] = prob[:, 1]
                cur_pred_val[i, :] = np.where(prob[:, 1] < thres, 0, 1)
                cur_mask = point_set_normalized_mask_np[i, :]
                cur_pred_prob_mask.append(cur_pred_prob[i, cur_mask])
                cur_pred_val_mask.append(cur_pred_val[i, cur_mask])
                target_mask.append(target_np[i, cur_mask])
            
            del cur_pred  # 删除不再需要的大数组

            if args.output:
                points_np = points.transpose(2, 1).cpu().numpy()
                pc_min_np = pc_min.numpy()
                pc_max_np = pc_max.numpy()

                for i in range(cur_batch_size):
                    cur_points = points_np[i, :, :]
                    cur_mask = point_set_normalized_mask_np[i, :]
                    cur_points = cur_points[cur_mask, :]
                    
                    output_points = np.zeros((cur_points.shape[0], 4), dtype=np.float64)
                    output_points[:, 0:2] = cur_points[:, 0:2]
                    cur_pc_min = pc_min_np[i, :]
                    cur_pc_max = pc_max_np[i, :]
                    output_points[:, 0:2] = pc_denormalize(output_points[:, 0:2], cur_pc_min, cur_pc_max)

                    output_points[:, 2] = cur_pred_prob_mask[i]
                    output_points[:, 3] = cur_pred_val_mask[i]

                    output_file = os.path.basename(fn[i])
                    output_path = os.path.join(output_dir, output_file)
                    np.savetxt(output_path, output_points, delimiter=',', fmt='%.4f')
                    
                    # 计算每个文件的指标
                    segp = cur_pred_val_mask[i]
                    segl = target_mask[i]
                    
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        segl, segp, average='binary', pos_label=1, zero_division=0
                    )
                    
                    current_filename = os.path.basename(fn[i])
                    
                    # 实时打印每个文件的指标
                    print(f'Test Sample: {current_filename} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}')
                    log_string(f'Test Sample: {current_filename} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}')
                    
                    plot_points(global_index, cur_points, segp, 'test', current_filename, precision, recall, f1, args.exp_name)
                    global_index += 1
                    
                    results.append({
                        'Filename': current_filename,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1
                    })
                    
                    # 清理循环中的变量
                    del cur_points, output_points, segp, segl
                
                del points_np, pc_min_np, pc_max_np
            
            # 计算批处理的混淆矩阵
            # --- 在循环内部 ---
            # 将列表展平为一维数组
            target_mask_flat = np.concatenate(target_mask)
            cur_pred_val_mask_flat = np.concatenate(cur_pred_val_mask)
            
            # 强制使用 labels=[0, 1] 确保矩阵永远是 2x2，即使某个文件只有信号或只有噪声
            cm = confusion_matrix(target_mask_flat, cur_pred_val_mask_flat, labels=[0, 1])

            # 解包 (0是噪声，1是信号)
            # matrix layout:
            # [[TN (真噪声), FP (噪声误判为信号)],
            #  [FN (信号漏检), TP (真信号)]]
            tn, fp, fn, tp = cm.ravel()

            tp_acc += tp
            fp_acc += fp
            fn_acc += fn
            
            # 清理批处理变量
            del target_mask, cur_pred_val_mask, target_mask_flat, cur_pred_val_mask_flat, cm
            gc.collect()
            torch.cuda.empty_cache()

        # 计算并保存全局指标（基于文件级别的平均）
        results_df = pd.DataFrame(results)
        
        # 计算各文件指标的平均值
        avg_precision = results_df['Precision'].mean()
        avg_recall = results_df['Recall'].mean()
        avg_f1 = results_df['F1 Score'].mean()
        
        test_metrics['Precision'] = avg_precision
        test_metrics['Recall'] = avg_recall
        test_metrics['F1 score'] = avg_f1
        
        # 打印全局平均指标到控制台和日志
        # --- 在循环外部，计算 Global Metrics ---
        # 加 1e-6 是为了防止除以 0
        global_precision = tp_acc / (tp_acc + fp_acc + 1e-6)
        global_recall = tp_acc / (tp_acc + fn_acc + 1e-6)
        global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-6)

        print('\n' + '='*60)
        print('[Global Metrics (Weighted by Point Count)]') # 这一点更客观
        print(f'Precision: {global_precision:.5f}')
        print(f'Recall:    {global_recall:.5f}')
        print(f'F1 Score:  {global_f1:.5f}')
        print('='*60)

        # 保留原来的“算术平均”作为参考
        print('[File-level Arithmetic Mean (Macro Average)]')
        log_string('Precision (avg): %.5f' % test_metrics['Precision'])
        log_string('Recall (avg): %.5f' % test_metrics['Recall'])
        log_string('F1 score (avg): %.5f' % test_metrics['F1 score'])
        
        # 添加全局平均指标行
        global_metrics = {
            'Filename': '[Global Average]',
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1 Score': avg_f1
        }
        results_df = pd.concat([results_df, pd.DataFrame([global_metrics])], ignore_index=True)
        
        # 保存CSV文件，文件名使用实验名称
        if args.exp_name:
            csv_filename = f'{args.exp_name}.csv'
        else:
            csv_filename = 'per_file_metrics.csv'
        
        # 修改CSV保存路径到results目录
        if args.exp_name:
            csv_path = os.path.join(BASE_DIR, 'results', args.exp_name, csv_filename)
        else:
            csv_path = os.path.join(BASE_DIR, 'results', csv_filename)
        results_df.to_csv(csv_path, index=False)
        log_string(f'结果已保存至: {csv_filename}')

if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    finally:
        # 确保程序结束时清理所有资源
        torch.cuda.empty_cache()
        gc.collect()
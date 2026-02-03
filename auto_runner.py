import os
import subprocess
import time
import threading
from queue import Queue
from datetime import datetime

# ==============================================================================
# 1. 核心配置区域 (请根据你的硬件和需求修改此处)
# ==============================================================================

# [关键] 设置可用的 GPU ID 列表
# 脚本会自动利用这些 GPU 并行运行实验。
# 例如: ['0'] (单卡), ['0', '1'] (双卡), ['0', '1', '2', '3'] (四卡)
AVAILABLE_GPUS = ['0'] 

# 项目路径配置
SOURCEDIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SOURCEDIR, 'data')
SUMMARY_FILE = os.path.join(SOURCEDIR, 'all_experiments_summary.txt')

# 基础参数 (所有实验共用的默认参数)
BASE_CONFIG = {
    'split_file': 'train_val_test_split.py',
    'train_file': 'train_partseg.py',
    'test_file': 'test_partseg.py',
    
    # 训练参数
    'model': 'pointnet2_part_seg_msg', # 模型架构
    'epoch_num': 50,      # 建议设置为 100-200 以保证收敛
    'learning_rate': 0.001,
    'lr_decay': 0.9,
    'decay_rate': 1e-3,
    'step_size': 5,
    'loss_weight': 1.0,
    
    # 测试参数
    'test_npoint': 4096,   # 测试时的点数，None 则使用 npoint 的值
    'test_batch_size': 1,
    'threshold': 0.5,
    'num_votes': 10,       # 测试时的投票数
    'test_ckpt': 'model.pth',
    'skip_test': True,    # 如果为 True，则只训练不测试
}

# ==============================================================================
# 2. 消融实验列表 (在此处定义你想跑的所有实验组合)
# ==============================================================================
# 每一个字典代表一个实验，未指定的参数将使用 BASE_CONFIG 中的默认值
EXPERIMENT_LIST = [    
    # =================================================================
    # 组 1: 全量基准 (Baseline)
    # 极端测试：不丢弃任何信息，但 Batch Size=1 极其不稳定，用于设立“理论上限”
    # =================================================================
    {
        'name': 'Exp01_npoint[None]_batchsize[1]',
        'loss_type': 'adaptive_focal',
        'npoint': None,        
        'batch_size': 1,       
    },

    # =================================================================
    # 组 2: 高保真组 (High Fidelity)
    # 覆盖最大文件 (55k)，Padding 到 64k。
    # 显存压力巨大，Batch Size 只能很小。
    # =================================================================
    {
        'name': 'Exp02_npoint[65536]_batchsize[2]',
        'loss_type': 'adaptive_focal',
        'npoint': 65536,
        'batch_size': 1,       
    },
    {
        'name': 'Exp03_npoint[32768]_batchsize[4]',
        'loss_type': 'adaptive_focal',
        'npoint': 32768,       # 约为最大文件的一半
        'batch_size': 2,       
    },

    # =================================================================
    # 组 3: 统计平均组 (Statistical Average) - 重点关注
    # 16384 最接近你的平均点数 (15003)，理论上是信息与效率的最佳平衡点。
    # =================================================================
    {
        'name': 'Exp04_npoint[16384]_batchsize[8]',
        'loss_type': 'adaptive_focal',
        'npoint': 16384,
        'batch_size': 4,       
    },

    # =================================================================
    # 组 4: 效率优先组 (High Efficiency / Max Batch Size)
    # [补齐部分]：从这里开始，点数减少，Batch Size 达到上限 16。
    # 这部分旨在验证：是否牺牲一部分点云细节，换取更稳定的 BN 层统计(BS=16)，能得到更好的结果？
    # =================================================================
    {
        'name': 'Exp05_npoint[8192]_batchsize[16]',
        'loss_type': 'adaptive_focal',
        'npoint': 8192,        # [补齐] 16k的一半，通常跑得很快且效果不错
        'batch_size': 8,      # 达到你设定的最大 BS
    },
    {
        'name': 'Exp06_npoint[4096]_batchsize[16]',
        'loss_type': 'adaptive_focal',
        'npoint': 4096,        # [补齐] 进一步降低分辨率
        'batch_size': 12,      
    },
    {
        'name': 'Exp07_npoint[2048]_batchsize[16]',
        'loss_type': 'adaptive_focal',
        'npoint': 2048,        # [修正] 你的最小文件是559，2048是一个比较安全的下界
        'batch_size': 16,       
    },

    # # --- 第二组：点云密度与Batch Size实验 ---
    # # 注意：点数减少时，可以适当增大 Batch Size 以加快速度
    # {
    #     'name': 'Exp04_Points_2048',
    #     'loss_type': 'adaptive_focal',
    #     'npoint': 2048,
    #     'batch_size': 8, 
    # },
    # {
    #     'name': 'Exp05_Points_1024',
    #     'loss_type': 'adaptive_focal',
    #     'npoint': 1024,
    #     'batch_size': 16,
    # },
    
    # # --- 第三组：学习率敏感性实验 ---
    # {
    #     'name': 'Exp06_LR_High',
    #     'learning_rate': 0.005,
    #     'loss_type': 'adaptive_focal'
    # },
    #  {
    #     'name': 'Exp07_LR_Low',
    #     'learning_rate': 0.0001,
    #     'loss_type': 'adaptive_focal'
    # },
]

# ==============================================================================
# 3. 基础设施类 (无需修改)
# ==============================================================================

# 初始化 GPU 资源池
gpu_queue = Queue()
for gpu in AVAILABLE_GPUS:
    gpu_queue.put(gpu)

# 线程锁，防止写入文件冲突
file_write_lock = threading.Lock()

def write_summary(message, to_console=True):
    """将信息写入汇总 TXT 文件，并选择性打印到控制台"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    
    with file_write_lock:
        with open(SUMMARY_FILE, 'a', encoding='utf-8') as f:
            f.write(full_msg + "\n")
    
    if to_console:
        print(full_msg)

def run_single_experiment(exp_config):
    """
    单个实验的完整执行逻辑（运行在独立线程中）
    """
    # 1. 申请 GPU 资源 (如果队列为空，此处会阻塞等待)
    gpu_id = gpu_queue.get()
    
    exp_name = exp_config['name']
    
    # 合并配置
    config = BASE_CONFIG.copy()
    config.update(exp_config)
    
    # 该实验的详细日志文件
    exp_log_file = os.path.join(SOURCEDIR, f"{exp_name}_details.log")
    
    write_summary(f">>> [启动] 实验: {exp_name} | GPU: {gpu_id}")

    try:
        # ==================== 阶段 2.1: 训练模型 ====================
        
        # 检查模型是否已存在
        model_checkpoint_path = os.path.join(SOURCEDIR, 'log', 'part_seg', exp_name, 'checkpoints', 'model.pth')
        model_exists = os.path.exists(model_checkpoint_path)
        
        if model_exists:
            write_summary(f"    检测到已有模型: {model_checkpoint_path}，跳过训练，直接执行测试")
        else:
            write_summary(f"    未检测到模型，开始训练...")
            
            train_cmd = [
                'python', config['train_file'],
                '--model', config['model'],
                '--epoch', str(config['epoch_num']),
                '--batch_size', str(config['batch_size']),
                '--learning_rate', str(config['learning_rate']),
                '--data_root', DATA_ROOT,
                '--conf', # 假设这是一个布尔flag
                '--loss_weight', str(config['loss_weight']),
                '--lr_decay', str(config['lr_decay']),
                '--step_size', str(config['step_size']),
                '--decay_rate', str(config['decay_rate']),
                '--gpu', gpu_id,
                '--log_dir_name', exp_name,  # 强制指定日志目录名，方便后续测试查找
                '--loss_type', config.get('loss_type', 'nll')
            ]

            # 添加条件参数
            if 'focal_gamma' in config:
                train_cmd.extend(['--focal_gamma', str(config['focal_gamma'])])
            if config.get('npoint') is not None:
                train_cmd.extend(['--npoint', str(config['npoint'])])
            
            if config.get('ckpt'):
                train_cmd.extend(['--ckpt', config['ckpt']])

            # 执行训练
            with open(exp_log_file, 'w') as log_f:
                log_f.write(f"========== Training Start: {datetime.now()} ==========\n")
                log_f.flush()
                
                # 运行命令，将 stdout 和 stderr 都重定向到 log 文件
                process = subprocess.run(train_cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=SOURCEDIR)
                
                if process.returncode != 0:
                    raise Exception("训练脚本返回非零状态码，请查看详细日志。")

            write_summary(f"--- [训练完成] 实验: {exp_name}")

        # ==================== 阶段 2.2: 测试模型 ====================
        
        # 检查是否跳过测试
        if config.get('skip_test', False):
            write_summary(f"### [跳过测试] 实验: {exp_name} | 根据配置跳过测试阶段")
            return
        
        # 确定训练生成的日志目录路径
        # 直接使用实验名称，test_partseg.py 会自动拼接完整路径
        trained_log_dir = exp_name
        
        test_cmd = [
            'python', config['test_file'],
            '--batch_size', str(config['test_batch_size']),
            '--log_dir', trained_log_dir,  # 指向刚才训练好的目录
            '--data_root', DATA_ROOT,
            '--conf',
            '--ckpt', config['test_ckpt'],
            '--threshold', str(config['threshold']),
            '--num_votes', str(config['num_votes']),
            '--gpu', gpu_id,
            '--exp_name', exp_name  # 传递实验名称用于CSV文件命名
        ]

        # 测试时优先使用 test_npoint
        # 如果配置中明确设置了 test_npoint（包括设为None），就使用该值
        # 只有当配置中没有 test_npoint 键时，才回退到使用 npoint
        if 'test_npoint' in config:
            test_npoint = config['test_npoint']
        else:
            test_npoint = config.get('npoint')
        
        # 只有当 test_npoint 不为 None 时才添加 --num_point 参数
        if test_npoint is not None:
            test_cmd.extend(['--num_point', str(test_npoint)])

        # 执行测试
        with open(exp_log_file, 'a') as log_f: # 追加模式
            log_f.write(f"\n\n========== Testing Start: {datetime.now()} ==========\n")
            log_f.flush()
            
            process = subprocess.run(test_cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=SOURCEDIR)
            
            if process.returncode != 0:
                raise Exception("测试脚本返回非零状态码。")

        # ==================== 阶段 3: 结果提取 ====================
        # 尝试从日志文件的最后几行中提取 Accuracy 或 mIoU
        final_result = "结果已存入日志，未自动识别数值"
        try:
            with open(exp_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # 倒序查找最后 30 行
                for line in reversed(lines[-30:]):
                    line_lower = line.lower()
                    # 根据常见的输出格式查找关键字
                    if any(x in line_lower for x in ['iou', 'accuracy', 'acc', 'overall']):
                        final_result = line.strip()
                        break
        except Exception:
            pass

        write_summary(f"### [全部完成] 实验: {exp_name} | 结果摘要: {final_result}")

    except Exception as e:
        write_summary(f"!!! [异常失败] 实验: {exp_name} | 错误信息: {str(e)}")
    
    finally:
        # 关键：无论成功失败，都必须释放 GPU 资源，让队列中的下一个任务运行
        gpu_queue.put(gpu_id)
        gpu_queue.task_done()
        print(f"[资源释放] GPU {gpu_id} 已准备好接收新任务")

# ==============================================================================
# 4. 主程序入口
# ==============================================================================

def main():
    os.chdir(SOURCEDIR)
    
    # 1. 初始化汇总文件
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(f"实验汇总报告 - 开始时间: {datetime.now()}\n")
        f.write(f"可用 GPU: {AVAILABLE_GPUS}\n")
        f.write("=" * 80 + "\n\n")

    print(f"检测到可用 GPU: {AVAILABLE_GPUS}")
    print(f"计划运行实验总数: {len(EXPERIMENT_LIST)}")
    print(f"汇总文件路径: {SUMMARY_FILE}")

    # 2. 数据分割 (只需运行一次)
    print("\n[步骤 1/2] 正在检查/分割训练数据...")
    split_cmd = ['python', BASE_CONFIG['split_file'], '--data_dir', DATA_ROOT]
    # 简单的运行一次 split，如果失败则退出
    if subprocess.run(split_cmd).returncode != 0:
        print("错误：数据分割脚本执行失败，程序终止。")
        return
    print("数据分割准备就绪。")

    # 3. 启动多线程实验
    print("\n[步骤 2/2] 开始并发执行实验...")
    threads = []
    
    for exp_config in EXPERIMENT_LIST:
        # 为每个实验配置创建一个线程
        t = threading.Thread(target=run_single_experiment, args=(exp_config,))
        t.start()
        threads.append(t)
        
        # 稍微延迟一下，避免所有线程同时瞬间启动造成的日志打印混乱
        time.sleep(2)

    # 4. 等待所有实验结束
    for t in threads:
        t.join()

    print("\n" + "=" * 60)
    print(f"所有消融实验执行完毕！")
    print(f"请查看汇总文件以获取结果: {SUMMARY_FILE}")
    print("=" * 60)

if __name__ == '__main__':
    main()
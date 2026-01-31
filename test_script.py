import subprocess
import os

# 自动获取项目根目录（test_script.py所在目录）
sourcedir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(sourcedir, 'data')
test_file = 'test_partseg.py'
npoint = 10000  # 将 npoint 设为 None，表示不固定点云数量
batch_size = 1
threshold = 0.5  # 如果预测的概率大于 50%，则判断该点属于目标类，否则属于背景类或其他类别
num_votes = 10
# 每次都要改为最新的-------------------------------------------------------------------！！！
log_dir = '2025-05-08'  
ckpt = 'model.pth'  # 或者 'ckpt_500.pth'，也就是最后整50轮的训练结果， 'model.pth' 就是目前最新的轮次的模型，可以根据需要设置

# 更改工作目录
os.chdir(sourcedir)

# 构建命令
test_command = ['python', test_file,
                '--batch_size', str(batch_size),
                '--log_dir', log_dir,
                '--data_root', data_root,
                '--conf',
                '--ckpt', ckpt,
                '--threshold', str(threshold),
                '--num_votes', str(num_votes)]

# 如果 npoint 不为 None，则添加 --npoint 参数
if npoint is not None:
    test_command.extend(['--num_point', str(npoint)])

test_process = subprocess.Popen(test_command)
test_process.wait()  # 等待训练模型的进程结束

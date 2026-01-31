import os
import subprocess

# 自动获取项目根目录（train_script.py所在目录）
SOURCEDIR = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(SOURCEDIR, 'data')
split_file = 'train_val_test_split.py'
train_file = 'train_partseg.py'
model = 'pointnet2_part_seg_msg'
epoch_num = 500
npoint = None  # 将 npoint 设为 None，表示不固定点云数量
batch_size = 1 # 如果npoint为None，则batch_size只能为1，因为无法合并
learning_rate = 0.0001
lr_decay = 0.8 # 学习率衰减因子，通常用于在训练过程中逐渐降低学习率以提高模型的泛化能力。这里值为1表示没有进行学习率衰减
decay_rate = 1e-3
step_size = 20 # 通常与学习率调度器一起使用，表示经过多少个epoch后应用学习率衰减。与学习率衰减方案配合使用以更好地控制学习率调整
loss_weight = 1.0
ckpt = ''

# 进入指定目录
os.chdir(SOURCEDIR)

# 分割训练数据
split_process = subprocess.Popen(['python', split_file, '--data_dir', data_root])
split_process.wait()  # 等待分割数据的进程结束

# 训练模型
train_command = [
    'python', train_file,
    '--model', model,
    '--epoch', str(epoch_num),
    '--batch_size', str(batch_size),
    '--learning_rate', str(learning_rate),
    '--data_root', data_root,
    '--conf',
    '--loss_weight', str(loss_weight),
    '--lr_decay', str(lr_decay),
    '--ckpt', ckpt,
    '--step_size', str(step_size),
    # '--early_stopping',
    '--decay_rate', str(decay_rate)
]

# 如果 npoint 不为 None，则添加 --npoint 参数
if npoint is not None:
    train_command.extend(['--npoint', str(npoint)])

train_process = subprocess.Popen(train_command)
train_process.wait()  # 等待训练模型的进程结束


# # 好参数1
# SOURCEDIR = 'C:/Users/14711/Desktop/PointNet2-main'
# split_file = 'train_val_test_split.py'
# train_file = 'train_partseg.py'
# model = 'pointnet2_part_seg_msg'
# epoch_num = 50
# npoint = 8192  # 将 npoint 设为 None，表示不固定点云数量
# batch_size = 1 # 如果npoint为None，则batch_size只能为1，因为无法合并
# learning_rate = 0.001
# lr_decay = 0.9 # 学习率衰减因子，通常用于在训练过程中逐渐降低学习率以提高模型的泛化能力。这里值为1表示没有进行学习率衰减
# step_size = 1 # 通常与学习率调度器一起使用，表示经过多少个epoch后应用学习率衰减。与学习率衰减方案配合使用以更好地控制学习率调整
# loss_weight = 1.0
# data_root = 'C:/Users/14711/Desktop/PointNet2-main/data/'
# ckpt = ''

# # test结果
# Precision: 0.96665
# Recall: 0.99996
# F1 score: 0.98302
# eval IoU of part 0: 0.925714
# eval IoU of part 1: 0.999905
# Part avg mIOU is: 0.96281
# Val loss: 0.00087
# Val F1 score: 0.99988
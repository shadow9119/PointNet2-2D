"""
Author: Benny
Date: Nov 2019

Modified by Sitong Chen
Date: Oct 2024
"""
import argparse
import os, random
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import confusion_matrix
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 显示张量维度
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

# 绘制并保存二维点云分类结果的函数
import numpy as np
import matplotlib.pyplot as plt

def plot_points(index, points, pred_choice, dataset_name, save_path):
    """
    index: 当前批次的图像编号，用于命名图像文件。
    points: 当前批次的点云数据 (3, N)，N 是点云数量。
    pred_choice: 当前批次的预测标签 (N, )。
    dataset_name: 当前数据集的名称（如 'train'）。
    save_path: 图像保存的文件夹路径。
    """
    # 将点云数据和预测标签转换为 numpy 格式
    points_np = points.cpu().numpy().T  # 形状变为 (N, 3)
    pred_choice_np = pred_choice.cpu().numpy()  # (N,)

    # 检查 pred_choice 的长度是否与点云数量一致
    if len(pred_choice_np) != points_np.shape[0]:
        raise ValueError(f"预测标签的长度 ({len(pred_choice_np)}) 与点云数量 ({points_np.shape[0]}) 不一致")

    # 定义颜色映射
    color_map = {0: 'blue', 1: 'orange'}
    colors = [color_map[label] for label in pred_choice_np]

    # 绘制图像
    plt.figure(figsize=(10, 8))
    plt.scatter(points_np[:, 0], points_np[:, 1], c=colors, s=1, alpha=0.7)
    plt.title(f'{dataset_name.capitalize()} Set - Sample {index + 1}')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 添加图例
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Noise', markersize=5, markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='o', color='w', label='Signal', markersize=5, markerfacecolor='orange')
    ]
    plt.legend(handles=handles, title="Classes")

    # 保存图像
    plt.savefig(os.path.join(save_path, f'{dataset_name.capitalize()} Set-{index + 1}.svg'), bbox_inches='tight')
    plt.clf()
    plt.close()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    # if (y.is_cuda):
    #     return new_y.cuda()
    return new_y.to(y.device)
    # return new_y


def free_gpu_cache():
    torch.cuda.empty_cache()

# 设置参数，先按照train_script里的设置，未设置的再按这里的默认参数设置
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=None, help='point Number (default: None, use all points)')  # 修改为默认 None
    parser.add_argument('--conf', action='store_true', default=False, help='use confidence level')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--data_root', type=str, required=True, help='data root file')
    parser.add_argument('--loss_weight', type=float, default=1.0, help='training loss weight')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='use early stopping or not')

    return parser.parse_args()


def main(args):
    # 记录一条 INFO 级别的日志消息到配置的日志文件中，并将相同的消息打印到控制台。
    def log_string(str):
        logger.info(str)
        print(str)

    '''SEED SETTING'''
    # 确保在使用随机数时的可重复性，以及在使用 PyTorch的cudnn 进行深度学习时的确定性行为 → 保证结果可重复
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    '''GPU SETTING'''
    # 如果GPU可用则将程序运行在GPU上，否则在CPU上运行
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    # 创建文件夹 [./log/part_seg/日期(年月日时分秒)/checkpoints/] 和 [./log/part_seg/日期(年月日时分秒)/logs/]
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    pretrain_dir = Path('')
    if args.ckpt:
        pretrain_dir = exp_dir.joinpath(args.ckpt)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    # 在 [./log/part_seg/日期(年月日时分秒)/logs/]文件夹下，建立model名.txt的日志文件（比如pointnet2_part_seg_msg.txt)
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO) # 记录 INFO 级别及以上级别的日志消息（包括 WARNING、ERROR 和 CRITICAL）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...') # 同时打印到 model名.txt 日志文件和控制台
    log_string(args)

    # tensorboard set-up
    # 在 [./runs/日期(年月日时分秒)/]文件夹下，写入 TensorBoard 日志
    writer = SummaryWriter(os.path.join('runs', timestr))

    root = args.data_root  # 获得根目录路径

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='train', conf_channel=args.conf) # 创建训练数据集 TRAIN_DATASET，获取训练集的文件目录表
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=3, drop_last=True) # 在训练过程中按批次加载数据，batch_size指定每个批次的数据量
                                                                                    # shuffle=True 表示在每个 epoch 开始时对数据进行洗牌。
                                                                                    # num_workers=3 指定使用 3 个子进程来加载数据，以加速数据读取。
                                                                                    # drop_last=True 表示如果最后一个批次的样本不足一个批次的大小，则丢弃它。
    VAL_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='val', conf_channel=args.conf)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=3)
    log_string("train_partseg中，已读入train集个数: %d" % len(TRAIN_DATASET))
    log_string("train_partseg中，已读入val集个数:  %d" % len(VAL_DATASET))

    num_classes = 1
    num_part = 2

    early_stopping = args.early_stopping # 用于在训练过程中监控模型的性能，如果模型在验证集上的性能不再提升，则提前停止训练，防止过拟合和节省计算资源
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model) # 导入指定【深度学习】模型模块，将其赋值给 MODEL 变量，之后可以通过 MODEL 访问该模块中的函数和类
    # 复制模型相关文件到实验目录exp_dir
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, conf_channel=args.conf).to(device) # 实例化模型和损失函数 在models\pointnet2_part_seg_msg.py中
    # cross-entropy loss 交叉熵损失函数
    criterion = MODEL.get_loss().to(device)
    loss_weight = args.loss_weight
    weight = torch.Tensor([1.0, loss_weight]).to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data) # 使用 Xavier 正态分布来初始化权重，保持前向传播时激活值的均匀性，适用于 Sigmoid 和 Tanh 激活函数。m 表示某个神经网络层（通常是线性层或卷积层）
            torch.nn.init.constant_(m.bias.data, 0.0) # 将偏置初始化为常数 0。这是一个常用的做法，因为 Bias 通常不需要复杂的初始化
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(pretrain_dir))
        start_epoch = checkpoint['epoch']
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        classifier.load_state_dict(model_state_dict)
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    # 初始化优化器Adam
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(), # 获取模型参数
            lr=args.learning_rate, # 设置学习率
            betas=(0.9, 0.999), # 控制一阶矩（均值）和二阶矩（方差）的衰减率，通常使用 (0.9, 0.999)
            eps=1e-08, # 防止除零错误的一个小常数
            weight_decay=args.decay_rate # L2正则化的权重衰减
        )
    else: # SGD优化器
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9) # 使用动量momentum来加速 SGD 的收敛

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0

    train_loss_acc_list = []
    val_loss_acc_list = []
    train_f1_acc_list = []
    val_f1_acc_list = []

    loss_dict = {"train_loss": [], "val_loss": []}
    metric_dict = {"train_metric": [], "val_metric": []}

    for epoch in range(start_epoch, args.epoch+start_epoch):

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch+start_epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        loss_acc = []
        # f1_acc = []
        tp, fp, fn = 0, 0, 0

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:2] = provider.random_scale_point_cloud(points[:, :, 0:2])

            points = torch.Tensor(points)
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))  # seg_pred: probabilities after softmax

            # 确保输出的点云数量与输入一致
            seg_pred = seg_pred.contiguous().view(-1, num_part)  # seg_pred [BxN, num_part]
            target = target.view(-1, 1)[:, 0]  # target [BxN]
            pred_choice = seg_pred.data.max(1)[1]
            # max(1)返回沿轴1的最大值及其相应的索引
            # max(1)[0]返回最大值，max(1)[1]返回索引，在我们的例子中是类号

            ''' 绘制并保存训练集点云图像 '''
            # 设置保存路径
            save_path = os.path.join(BASE_DIR, 'results', 'train')
            os.makedirs(save_path, exist_ok=True)
            for j in range(points.shape[0]):  # 遍历每个样本
                # 计算样本在整个数据集中的索引
                global_index = i * args.batch_size + j
                if epoch == args.epoch - 1:
                    # 确保 pred_choice 的分割方式正确
                    start_idx = j * points.shape[2]  # points.shape[2] 是点云数量 N
                    end_idx = (j + 1) * points.shape[2]
                    plot_points(global_index, points[j], pred_choice[start_idx:end_idx], 'train', save_path)


            # calculate confusion metric  计算混淆度
            cm = confusion_matrix(pred_choice.detach(), target.detach(),task='binary', num_classes=num_part)
            cm = cm.cpu().numpy()
            #积累真阳性、假阳性和假阴性

            tp += cm[1, 1]
            fp += cm[0, 1]
            fn += cm[1, 0]

            loss = criterion(seg_pred, target, weight=weight)
            # loss = criterion(seg_pred, target)
            loss_acc.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        loss_acc = np.mean(loss_acc)
        train_loss_acc_list.append(loss_acc)
        log_string('Train loss: %.5f' % loss_acc)

        # calculate precision精度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        # calculate recall召回率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        # calculate f1 score F值
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        train_f1_acc_list.append(f1)
        log_string('Train F1 score: %.5f' % f1)

        val_loss_acc = []
        val_tp, val_fp, val_fn = 0, 0, 0

        # validation验证集
        with torch.no_grad():
            for i, (points, label, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader),
                                                      smoothing=0.9):
                points = points.data.numpy()

                points = torch.Tensor(points)
                points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                # seg_pred = seg_pred.contiguous().view(-1, 1)[:, 0]  # for sigmoid output
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]

                ''' 绘制并保存验证集点云图像 '''
                # 设置保存路径
                save_path = os.path.join(BASE_DIR, 'results', 'val')
                os.makedirs(save_path, exist_ok=True)
                for j in range(points.shape[0]):  # 遍历每个样本
                    # 计算样本在整个数据集中的索引
                    global_index = i * args.batch_size + j
                    if epoch == args.epoch - 1:
                        # 确保 pred_choice 的分割方式正确
                        start_idx = j * points.shape[2]  # points.shape[2] 是点云数量 N
                        end_idx = (j + 1) * points.shape[2]
                        plot_points(global_index, points[j], pred_choice[start_idx:end_idx], 'val', save_path)



                # calculate confusion metric
                cm = confusion_matrix(pred_choice, target, task='binary', num_classes=num_part)
                cm = cm.cpu().numpy()
                # accumulate true positives, false positives and false negatives
                val_tp += cm[1, 1]
                val_fp += cm[0, 1]
                val_fn += cm[1, 0]

                loss = criterion(seg_pred, target, weight=weight)
                # loss = criterion(seg_pred, target)
                val_loss_acc.append(loss.item())

        val_loss_acc = np.mean(val_loss_acc)
        val_loss_acc_list.append(val_loss_acc)
        log_string('Val loss: %.5f' % val_loss_acc)

        # calculate precision
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 1.0
        # calculate recall
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 1.0
        # calculate f1 score
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0
        val_f1_acc_list.append(val_f1)
        log_string('Val F1 score: %.5f' % val_f1)


        if epoch == start_epoch or (epoch + 1) % 10 == 0:
            # add scalar to tensorboard 向 TensorBoard 中添加标量
            writer.add_scalar('Learning rate', lr, epoch + 1)
            writer.add_scalar('Loss/train', loss_acc, epoch + 1)
            writer.add_scalar('Loss/val', val_loss_acc, epoch + 1)
            writer.add_scalar('F1 score/train', f1, epoch + 1)
            writer.add_scalar('F1 score/val', val_f1, epoch + 1)
            writer.flush()
            writer.close()

        # early stopping提前停止
        if early_stopping is True:
            if len(val_loss_acc_list) > 3 and np.all(np.abs(np.diff(val_loss_acc_list[-3:])) < 0.001):
                # 如果 val 损失在 3 个 epoch 内的改善小于 0.001，则保存模型
            # if len(val_loss_acc_list) > 3 and np.all(np.diff(val_loss_acc_list[-3:]) > 0):
            #     # 如果 val 损失在 3 个 epoch 内没有改善，则保存模型
                logger.info('Early Stopping...')
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

                free_gpu_cache()

                break

        # save checkpoints 保存检查点
        if (epoch + 1) % 50 == 0 and epoch != (args.epoch+start_epoch-1):
            logger.info('Save checkpoint at epoch %d...' % (epoch+1))
            save_ckptpath = str(checkpoints_dir) + '/ckpt_' + str(epoch + 1) + '.pth'
            log_string('Saving at %s' % save_ckptpath)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_ckptpath)

        global_epoch += 1

        free_gpu_cache()

    # 保存最佳模型
    logger.info('Save model...')
    savepath = str(checkpoints_dir) + '/model.pth' # 存储在trained_model/model.pth
    log_string('Saving at %s' % savepath)
    state = {
        'epoch': epoch + 1,     # 当前的训练轮次 (epoch从0开始)
        # 'train_acc': train_instance_acc,  # 训练集准确度
        # 'val_acc': val_instance_acc, # 验证集准确度
        'model_state_dict': classifier.state_dict(), # 模型的参数（权重）
        'optimizer_state_dict': optimizer.state_dict(), # 优化器的状态
    }
    torch.save(state, savepath) # 将 state 对象（包括模型参数、优化器状态）保存到指定路径 trained_model/model.pth 中

    # # 将模型输入测试集测试
    # logger.info('Test model...')
    # # run test
    # test_script = 'test_partseg.py'

    # # 构建基础命令
    # test_command = f'python {test_script} --batch_size {args.batch_size} --log_dir {timestr} --data_root {args.data_root}'

    # # 在 args.npoint 有效时添加 --num_point 参数
    # if args.npoint is not None:
    #     test_command += f' --num_point {args.npoint}'

    # # 判断--conf参数是否作为 test_partseg.py 的输入参数
    # if args.conf:
    #     test_command += ' --conf'

    # return_code = os.system(test_command)  # 运行 test_partseg.py 脚本，参数如上
    # if return_code != 0:  # 如果返回值不为 0，说明测试脚本 test_partseg.py 运行失败
    #     print("Run test script error")


if __name__ == '__main__':
    args = parse_args()
    main(args)

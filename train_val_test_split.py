'''
Created by Sitong Chen
Date: Mar 2025
'''
import os, json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# setting
parser = argparse.ArgumentParser(description='Train-val-test data split')
parser.add_argument('--data_dir', type=str, required=True, help='Input directory') # required=true: 代表不可省略

# 对每个数据文件夹，训练集80% 验证集10% 测试集10%；可在后续根据实际需要调整具体的使用分类
def split_data(file_list):
    train_list, tmp_list = train_test_split(file_list, test_size=0.2, random_state=42, shuffle=True)
    val_list, test_list = train_test_split(tmp_list, test_size=0.5, random_state=42, shuffle=True)
    print(len(train_list))
    print(len(val_list))
    print(len(test_list))

    return train_list, val_list, test_list

# 对simulated_code的处理
def collect_files_recursively(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):  # 假设文件是 .txt
                file_list.append(os.path.join(root, file))
    return file_list


def create_json_file(train_list, val_list, test_list, out_dir='data/train_val_test_split'):

    os.makedirs(out_dir, exist_ok=True)

    train_json = json.dumps(train_list) # 转换为 JSON 格式的字符串
    with open(os.path.join(out_dir, 'train_file_list.json'), 'w') as f:
        f.write(train_json)

    val_json = json.dumps(val_list)
    with open(os.path.join(out_dir, 'val_file_list.json'), 'w') as f:
        f.write(val_json)

    test_json = json.dumps(test_list)
    with open(os.path.join(out_dir, 'test_file_list.json'), 'w') as f:
        f.write(test_json)

def main(args):
    # 将所有数据文件分为四类：real_day, real_night, simulated_ICESat_2 和 simulated_code
    data_real_water = os.path.join(args.data_dir, 'real_water')
    data_real_day = os.path.join(args.data_dir, 'real_day')
    data_real_night = os.path.join(args.data_dir, 'real_night')
    data_simulated_ICESat_2 = os.path.join(args.data_dir, 'simulated_ICESat_2')
    data_simulated_code = os.path.join(args.data_dir, 'simulated_code')

    # 收集各个文件夹的文件
    file_list_real_water = collect_files_recursively(data_real_water)
    file_list_real_day = collect_files_recursively(data_real_day)
    file_list_real_night = collect_files_recursively(data_real_night)
    file_list_simulated_ICESat_2 = collect_files_recursively(data_simulated_ICESat_2)
    file_list_simulated_code = collect_files_recursively(data_simulated_code)

    # 以分层的方式将4个文件夹分成训练集train、验证集val和测试集test
    train_list_real_water,val_list_real_water,test_list_real_water=split_data(file_list_real_water)
    print("分割real_day集：")
    train_list_real_day, val_list_real_day, test_list_real_day = split_data(file_list_real_day)
    print("分割real_night集：")
    train_list_real_night, val_list_real_night, test_list_real_night = split_data(file_list_real_night)
    print("分割simulated_ICESat_2集：")
    train_list_simulated_ICESat_2, val_list_simulated_ICESat_2, test_list_simulated_ICESat_2 = split_data(file_list_simulated_ICESat_2)
    print("分割simulated_code集：")
    train_list_simulated_code, val_list_simulated_code, test_list_simulated_code = split_data(file_list_simulated_code)

    # 根据需要确定 train集 val集 和 test集
    train_all = train_list_simulated_code + val_list_simulated_code
    val_all = test_list_simulated_code
    test_all = train_list_real_water + val_list_real_water + test_list_real_water + train_list_real_day + val_list_real_day + test_list_real_day + train_list_real_night + val_list_real_night + test_list_real_night + train_list_simulated_ICESat_2 + val_list_simulated_ICESat_2 + test_list_simulated_ICESat_2

    create_json_file(train_all, val_all, test_all)
    print("train集总文件数：" + str(len(train_all)))
    print("val集总文件数：" + str(len(val_all)))
    print("test集总文件数：" + str(len(test_all)))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
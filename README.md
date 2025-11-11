# 2D_Photon_PointNet2-ENGLISH
A two-dimensional photon points denoising model based on PointNet++ using Pytorch

## User Manual
### Parameter Settings
- In the `train_script.py` file, set the absolute paths for `SOURCEDIR` and `data_root`, which are the root directory of the program and the root directory of the dataset, respectively.
- For the `npoint` parameter, setting it to `None` means using all points in the data as the original input, in which case `batch_size` must be 1; setting it to another integer means randomly sampling a fixed number of points from the data as the original input. If the number of points in the data is less than the fixed integer, random points will be added to make up the difference.

### Model Training
First, place the data set in the `data_root` path if necessary. For the selection of training set, verification set, and test set, adjust it in the `train_val_test_split.py` file. Then run:
```commandline
python ./train_script.py
```
- After the model is trained, a predict is performed directly with the test set.
### Model Predicting
First, place the data set in the `data_root` path if necessary. For the selection of training set, verification set, and test set, adjust it in the `train_val_test_split.py` file. Then run:
```commandline
python ./test_script.py
```
### Default data
- We provide `real_day`, `real_night`, `simulated_code` and `simulated_ICESat_2` subdatasets in the `./data` directory, which can be used directly for model training and prediction.
- A detailed description of the data set can be found in `data/simulated_code/README.md` and in our published article.

We also provide an example of a trained model, `Model.pth`, placed under the `./trained_model ` path.
## Published Paper 

## Acknowledgments
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
<br>
@article{,
      Author = {Yiwen lin, Anders Jensen Knudby},
      Title = {Global automated extraction of bathymetric photons from ICESat-2 data based on a PointNet++ model},
      Journal = {International Journal of Applied Earth Observation and Geoinformation},
      Year = {2023}
}


# 2D_Photon_PointNet2-中文
基于PointNet++的二维光子点去噪模型

## 使用说明
### 参数设置
- 在 `train_script.py` 文件里设置 `SOURCEDIR` 、`data_root` 的绝对路径，二者分别是程序根目录、数据集根目录
- 对 `npoint` 参数，设置为`None`时，表示将数据中所有点作为原始输入，此时 `batch_size` 必须为1；设置为其他整数时，表示将从数据中随机抽取固定整数个点作为原始输入，当数据中点的数量少于固定整数时，会随机增加点进行补齐。

### 模型训练
首先，根据需要在`data_root`路径下放置数据集，对于训练集、验证集、测试集的选择，请在 `train_val_test_split.py` 文件中调整。然后运行：
```commandline
python ./train_script.py
```
- 模型训练完成后，会直接利用测试集进行一次测试。
### 模型预测
首先，根据需要在`data_root`路径下放置数据集，对于训练集、验证集、测试集的选择，请在 `train_val_test_split.py` 文件中调整。然后运行：
```commandline
python ./test_script.py
```
### 默认数据
- 我们在 `./data` 目录下提供了 `real_day`、`real_night`、`simulated_code` 和 `simulated_ICESat_2` 四个子数据集，可以直接用于模型训练和预测。
- 关于数据集的详细介绍可以参考 `data/simulated_code/README.md` 以及我们发表的文章。

我们也提供了训练好的模型示例 `model.pth`，放置在 `./trained_model` 路径下。

### 发表文章

### 致谢
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
<br>
@article{,
      Author = {Yiwen lin, Anders Jensen Knudby},
      Title = {Global automated extraction of bathymetric photons from ICESat-2 data based on a PointNet++ model},
      Journal = {International Journal of Applied Earth Observation and Geoinformation},
      Year = {2023}
}

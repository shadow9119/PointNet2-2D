# 数据集结构说明

数据文件均为 txt 格式，共有四列，分别为：沿轨距离 (x)，高程 (z)，点序号 (index)，类别 (1：信号，0：噪声)。

## 一级子文件夹

- **30μJ**: ICESat-2激光器能量-弱
- **120μJ**: ICESat-2激光器能量-强

## 二级子文件夹

- **noiseRate_0p5**: 噪声率 0.5 MHz
- **noiseRate_2**: 噪声率 2 MHz
- **noiseRate_5**: 噪声率 5 MHz

## 三级子文件夹

- **plane**: 平面地形
  - 命名格式：`plane_{elevation}`，其中 `{elevation}` 是平面高程。

- **inclined_plane**: 斜面地形
  - 命名格式：`inclined_plane_{slope}`，
其中 `{slope}` 为坡度（单位：度），斜面起始高程为 0。

- **ladder**: 阶梯地形
  - 命名格式：`ladder_n{steps}_w{stepwidth}_h{stepheight}`，其中 `{steps}` 是阶梯个数， `{stepwidth}` 是阶梯宽度，`{stepheight}`是阶梯高度，阶梯起始高程为 0。

- **plain**: 平原地形
  - 命名格式：`plain_{elevation}_f{noise_amplitude}`，
其中第一个 `{elevation}` 为平原平均高程，第二个 `{noise_amplitude}` 为噪声幅度。
  - 不同高程的噪声幅度有所不同，模拟更丰富的平原地形波动。

- **slope**: 斜坡地形
  - 命名格式：`slope_{slope}_f{noise_amplitude}`，
第一个 `{slope}` 为坡度（单位：度），第二个 `{noise_amplitude}` 为噪声幅度。
  - 斜坡起始高程为 0，不同高程的噪声幅度有所不同，模拟更丰富的斜坡地形波动。

- **mountain**: 山脉地形
  - 命名格式：`mountain_{elevation}_f{noise1}_{noise2}_{noise3}`，
其中第一个 `{elevation}` 为山脉平均高程，其余 `{noise1}`, `{noise2}`, `{noise3}` 依次为三个不同的 Perlin 噪声幅度。
  - 不同高程的噪声幅度有所不同，模拟更丰富的山脉地形波动。

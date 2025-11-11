# Dataset Structure Description

The data files are in txt format, with four columns: track distance (x), elevation (z), point index (index), category (1: signal, 0: noise).

## Primary Subfolders

- **noiseRate_0p5**: Noise rate 0.5 MHz
- **noiseRate_2**: Noise rate 2 MHz
- **noiseRate_5**: Noise rate 5 MHz

## Secondary Subfolders

- **plane**: Flat terrain
  - Naming format: `plane_{elevation}`, where `{elevation}` is the elevation of the plane.

- **inclined_plane**: Inclined terrain
  - Naming format: `inclined_plane_{slope}`,
where `{slope}` is the slope angle (in degrees), with the starting elevation of 0.

- **ladder**: Ladder terrain
  - Naming format: `ladder_{steps}`, where `{steps}` is the number of steps, with the number of steps * step width ≈ 5000 (maximum track distance).

- **plain**: Plain terrain
  - Naming format: `plain_{elevation}_{noise_amplitude}`,
where the first `{elevation}` is the average elevation of the plain, and the second `{noise_amplitude}` is the noise amplitude.
  - Different elevations have different noise amplitudes, simulating more varied plain terrain undulations.

- **slope**: Slope terrain
  - Naming format: `slope_{slope}_{noise_amplitude}`,
where the first `{slope}` is the slope, and the second `{noise_amplitude}` is the noise amplitude.
  - The slope starts at an elevation of 0, and different elevations have different noise amplitudes, simulating more varied slope terrain undulations.

- **mountain**: Mountain terrain
  - Naming format: `mountain_{elevation}_{noise1}_{noise2}_{noise3}`,
where the first `{elevation}` is the average elevation of the mountain, and the remaining `{noise1}`, `{noise2}`, `{noise3}` are three different Perlin noise amplitudes.
  - Different elevations have different noise amplitudes, simulating more varied mountain terrain undulations.  

<br>

# 数据集结构说明

数据文件均为 txt 格式，共有四列，分别为：沿轨距离 (x)，高程 (z)，点序号 (index)，类别 (1：信号，0：噪声)。

## 一级子文件夹

- **noiseRate_0p5**: 噪声率 0.5 MHz
- **noiseRate_2**: 噪声率 2 MHz
- **noiseRate_5**: 噪声率 5 MHz

## 二级子文件夹

- **plane**: 平面地形
  - 命名格式：`plane_{elevation}`，其中 `{elevation}` 是平面高程。

- **inclined_plane**: 斜面地形
  - 命名格式：`inclined_plane_{slope}`，
其中 `{slope}` 为坡度（单位：度），斜面起始高程为 0。

- **ladder**: 阶梯地形
  - 命名格式：`ladder_{steps}`，其中 `{steps}` 是阶梯个数，阶梯个数 * 阶梯宽度 ≈ 5000（沿轨距离最大值）。

- **plain**: 平原地形
  - 命名格式：`plain_{elevation}_{noise_amplitude}`，
其中第一个 `{elevation}` 为平原平均高程，第二个 `{noise_amplitude}` 为噪声幅度。
  - 不同高程的噪声幅度有所不同，模拟更丰富的平原地形波动。

- **slope**: 斜坡地形
  - 命名格式：`slope_{slope}_{noise_amplitude}`，
第一个 `{slope}` 为坡度，第二个 `{noise_amplitude}` 为噪声幅度。
  - 斜坡起始高程为 0，不同高程的噪声幅度有所不同，模拟更丰富的斜坡地形波动。

- **mountain**: 山脉地形
  - 命名格式：`mountain_{elevation}_{noise1}_{noise2}_{noise3}`，
其中第一个 `{elevation}` 为山脉平均高程，其余 `{noise1}`, `{noise2}`, `{noise3}` 依次为三个不同的 Perlin 噪声幅度。
  - 不同高程的噪声幅度有所不同，模拟更丰富的山脉地形波动。

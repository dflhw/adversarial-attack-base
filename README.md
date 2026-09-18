# adversarial-attack-base

ImageNet 白盒对抗样本实验脚本集合。每个文件对应一种迭代攻击算法，流程相同：读图 → 加载 torchvision 预训练分类器 → 在 ε 球内更新像素 → 将「原图 / 对抗图 / 扰动」拼成一张图保存，并统计攻击成功率。

## 目录

```
.
├── bim_ilcm.py      # BIM / ILCM
├── pgd.py           # PGD（随机初始化 + 多次重启）
├── mim.py           # MIM（动量迭代）
├── nim.py           # NIM（Nesterov 动量）
├── mtpgd.py         # Multi-Target PGD
├── odipgd.py        # ODI-PGD
├── labels.txt       # 测试集：文件名 + ImageNet 类别 ID
└── test_images/     # 50 张 ILSVRC2012 验证集样例
```

## 环境

依赖：

- Python 3
- PyTorch、torchvision
- numpy、Pillow、tqdm
- CUDA（可选；无 GPU 时会回退到 CPU）

安装示例：

```bash
pip install torch torchvision numpy pillow tqdm
```

分类器通过 `torchvision.models.<name>(pretrained=True)` 加载，`--classifier` 可填 `vgg13`、`vgg16`、`resnet50` 等 torchvision 中的模型名。首次运行会下载对应预训练权重。

> 脚本使用了 `torch.autograd.gradcheck.zero_gradients`，该接口在较新的 PyTorch 中已移除。若报错，请改用仍提供该函数的旧版 PyTorch，或自行将调用替换为 `tensor.grad = None`。

## 数据格式

`labels.txt` 每行：`文件名 类别ID`

```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
```

除 `bim_ilcm.py` 外，图像目录和标注文件分开传入：

```bash
--img_dir test_images --img_txt_dir labels.txt
```

`bim_ilcm.py` 约定根目录下同时有标注文件和 `dataset/` 子目录：

```
<root>/
  ├── <txt_name>          # 默认 resnet50_2000.txt
  └── dataset/
      └── *.JPEG
```

默认路径写在原作者机器上（`/home/guest/liu_hanwen/...`），本地运行请显式传入数据路径。

## 运行

`batch_size` 仅支持 1。`--target` 为 `argparse` 的 `bool` 类型：命令行传入任意非空字符串都会变成 `True`（包括 `--target False`）。无目标攻击请省略该参数，沿用脚本默认值。

### PGD

```bash
python pgd.py \
  --img_dir test_images \
  --img_txt_dir labels.txt \
  --output_dir output/pgd \
  --classifier vgg13 \
  --GPU 0
```

默认执行有目标攻击，目标类取模型对原图 logits 最小的类别。

### BIM / ILCM

```bash
python bim_ilcm.py \
  --root /path/to/project_dataset \
  --txt_name resnet50_2000.txt \
  --output_dir output/bim_ilcm \
  --classifier resnet50 \
  --GPU 0
```

无随机重启。默认无目标；有目标时同样攻击 logits 最小类。

### MIM

```bash
python mim.py \
  --img_dir test_images \
  --img_txt_dir labels.txt \
  --output_dir output/mim \
  --classifier vgg16 \
  --momentum 1.0 \
  --GPU 0
```

无目标。梯度先 L1 归一化，再按动量累积后做符号更新。

### NIM

```bash
python nim.py \
  --img_dir test_images \
  --img_txt_dir labels.txt \
  --output_dir output/nim \
  --classifier vgg16 \
  --momentum 1.0 \
  --GPU 0
```

无目标。在当前点沿动量方向前瞻一步，再计算梯度。

### Multi-Target PGD

```bash
python mtpgd.py \
  --img_dir test_images \
  --img_txt_dir labels.txt \
  --output_dir output/mtpgd \
  --classifier vgg16 \
  --sel_cl_num 10 \
  --restart_number 10 \
  --GPU 0
```

对原图 logits 最高的 `--sel_cl_num` 个类别分别做 PGD，任一目标使预测偏离原标签即视为成功。

### ODI-PGD

```bash
python odipgd.py \
  --img_dir test_images \
  --img_txt_dir labels.txt \
  --output_dir output/odipgd \
  --classifier vgg16 \
  --GPU 0
```

每次重启先沿随机输出方向走若干步（默认 4 步，步长 0.02），再切换为普通 PGD。默认有目标。

## 常用参数

| 参数 | 含义 | 常见默认值 |
|------|------|------------|
| `--imgsize` | 输入分辨率 | 224 |
| `--mean` / `--std` | 归一化 | 0.5 / 0.5 |
| `--eps` | 扰动半径（归一化空间） | 0.03125（约 8/256） |
| `--step_size` | 单步步长 | 0.005 |
| `--max_epoch` | 每次重启的最大迭代次数 | 40 或 100 |
| `--restart_number` | 随机重启次数 | 10 或 100 |
| `--classifier` | torchvision 模型名 | 见上表 |
| `--GPU` | CUDA 设备编号 | `0` 或 `2` |
| `--seed` | 随机种子 | 7923 |
| `--output_dir` | 成功样本保存目录 | 需自行指定 |

扰动在像素归一化到 `[-1, 1]` 后的空间中裁剪：`[img - 2ε, img + 2ε]`，并再夹到 `[-1, 1]`。

## 输出

每张原图若分类正确，才会进入攻击。成功后保存一张宽为 `3 × imgsize` 的拼图：左原图、中对抗图、右绝对扰动。终端会打印分类错误率和攻击成功率（分母为原模型分类正确的样本数）。

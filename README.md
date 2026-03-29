# 🧬 biovision-lab

**biovision-lab** 是一个专注于生物医学成像（Biomedical Imaging）的深度学习实验仓库。本项目旨在探索计算机视觉技术（尤其是目标检测与分割算法）在医学诊断中的应用，涵盖从人脑/动物脑 MRI 分析到血管系统自动识别的多种场景。

---

## 🚀 核心目标

- **多模态检测**：利用 YOLO 系列（v8/v10）及 RT-DETR 实现医学影像病灶的快速定位。
- **跨物种分析**：探索模型在人脑与动物脑（如鼠、灵长类）MRI 图像上的泛化与辨别能力。
- **精准诊断**：集成 MONAI 框架，进行解剖结构的精准分割与定量评价。
- **工程实践**：遵循低耦合设计，支持从数据预处理（DICOM/NIfTI）到模型部署的全流程。

---

## 📂 目录结构

```
biovision-lab/
├── data/               # 数据集管理 (Raw/Processed)
├── deployments/        # 模型导出 (ONNX/TensorRT)
├── experiments/        # 实验记录与 Jupyter Notebooks
├── src/                # 核心源代码
│   ├── core/           # 模型架构定义 (YOLO, U-Net, etc.)
│   ├── data_loader/    # 医学影像专用读取器 (Dicom/NIfTI)
│   ├── loss/           # 针对医学不平衡数据的损失函数 (Dice, Focal)
│   └── utils/          # 可视化与评价指标 (mAP, Dice Score)
├── configs/            # 训练超参数配置文件 (YAML)
├── requirements.txt    # 依赖项
└── train.py            # 统一训练入口
```

---

## 🛠️ 技术栈

| 类别 | 工具 |
|---|---|
| **语言** | Python 3.9+ |
| **深度学习** | PyTorch, Ultralytics (YOLOv8/v10) |
| **医学专用** | MONAI (Medical Open Network for AI), Nibabel, Pydicom |
| **图像处理** | OpenCV, Scikit-image, Albumentations |

---

## 🧪 正在进行的实验

- [ ] **Stroke-Detection**: 基于脑部 MRI 的脑卒中区域快速筛查。
- [ ] **Vessel-Extractor**: 血管扫描影像的拓扑结构提取。
- [ ] **Cross-Species-MRI**: 人脑与实验动物脑部形态学的自动辨别实验。

---

## 🏗️ 快速开始

```bash
# 克隆仓库
git clone https://github.com/lynx009/biovision-lab.git
cd biovision-lab

# 创建环境
conda create -n biovision python=3.9
conda activate biovision

# 安装依赖
pip install -r requirements.txt
```

---

## 📖 使用示例

```bash
# 训练模型
python train.py --config configs/train.yaml

# 指定任务类型
python train.py --config configs/train.yaml --task detection
python train.py --config configs/train.yaml --task segmentation
```

---

## 📄 许可证

本项目基于 [LICENSE](LICENSE) 发布。

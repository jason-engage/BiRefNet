# Enhanced BiRefNet Training Pipeline

> **Comprehensive monitoring, logging, and experiment tracking**

## 🚀 Training Enhancements

This fork adds production-ready training infrastructure to the excellent BiRefNet model for high-resolution dichotomous image segmentation.

### ✨ Key Features

#### 📊 **Advanced Monitoring & Logging**
- **Gradient norm tracking** - Monitor training stability in real-time
- **Color-coded performance indicators** - Instant visual feedback on model performance
- **Comprehensive validation metrics** - Best/worst sample tracking, Jaccard distribution analysis
- **Dataset composition tracking** - Monitor multi-dataset training balance
- **tqdm progress bars** - Clean progress visualization for validation

#### 🔬 **Experiment Management**
- **Comet ML integration** - Full experiment tracking, comparison, and visualization
- **External configuration** - `config_vars.yml` for easy parameter management without code changes
- **Automatic checkpoint naming** - Organized saves with experiment names and epochs
- **Secure API key management** - Template-based config with gitignored sensitive data

#### 🎯 **Training Optimizations**
- **Dynamic batch sizing** - Automatic size adjustment with performance optimization
- **Support for .bin checkpoints** - Load HuggingFace format weights
- **Ranger optimizer support** - Alternative optimization strategy
- **Enhanced preprocessing** - Added `random_rotate_zoom` augmentation
- **Smart resume** - Automatic epoch detection from checkpoint filenames

#### 🛠️ **Developer Experience**
- **VS Code settings included** - Consistent formatting across the team
- **Comprehensive CLAUDE.md** - AI assistant instructions for codebase
- **Clean configuration template** - Get started quickly with sensible defaults

---

## 📋 Quick Start

### 1. Setup Configuration

```bash
# Copy the configuration template
cp config_vars_template.yml config_vars.yml

# Edit with your settings
nano config_vars.yml
```

### 2. Configure Comet ML (Optional but Recommended)

1. Sign up for free at [Comet.com](https://www.comet.com/)
2. Get your API key from [settings](https://www.comet.com/api/my/settings)
3. Update in `config_vars.yml`:
   ```yaml
   comet_ml_enable: true
   comet_ml_api_key: "YOUR_API_KEY"
   comet_ml_workspace: "YOUR_WORKSPACE"
   ```

### 3. Start Training

```bash
# Single GPU
python train.py --experiment_name my_experiment

# Multi-GPU with DDP
torchrun --nproc_per_node=4 train.py --dist True --experiment_name my_experiment

# With Accelerate (FP16/BF16)
accelerate launch train.py --use_accelerate --experiment_name my_experiment
```

## 📈 Training Output Example

```
E50 - 100/200 - L: 0.234, LB: 0.198, BC: 0.998, BCB: 0.999, DI: 0.985, DIB: 0.991,
JI: 0.971, JIB: 0.983, SI: 0.945, SIB: 0.962, LR: 0.0000071, M: 0.90, GN: 4.2,
BC: 2\0\1\0\0\0, TBC: 200\0\100\0\0\0, It/s: 3.2, ETA: 0:00:31
```

**Legend:**
- L/LB: Loss (epoch avg/batch)
- BC/BCB: BCE (avg/batch)
- DI/DIB: Dice (avg/batch)
- JI/JIB: Jaccard IoU (avg/batch)
- SI/SIB: SSIM (avg/batch)
- GN: Gradient Norm
- BC/TBC: Batch/Total composition per dataset

## 📊 Validation Features

- **Performance distribution visualization**
- **Top 10 best/worst performing samples**
- **Jaccard histogram with bucketing**
- **Real-time metric updates with tqdm**
- **Automatic sample image logging to Comet ML**

---

## 🎓 Original BiRefNet

<h3 align="center">Bilateral Reference for High-Resolution Dichotomous Image Segmentation</h3>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=TZRzWOsAAAAJ' target='_blank'><strong>Peng Zheng</strong></a><sup> 1,4,5,6</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=0uPb8MMAAAAJ' target='_blank'><strong>Dehong Gao</strong></a><sup> 2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=kakwJ5QAAAAJ' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 1*</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=9cMQrVsAAAAJ' target='_blank'><strong>Li Liu</strong></a><sup> 3</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=qQP6WXIAAAAJ' target='_blank'><strong>Jorma Laaksonen</strong></a><sup> 4</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=pw_0Z_UAAAAJ' target='_blank'><strong>Wanli Ouyang</strong></a><sup> 5</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=stFCYOAAAAAJ' target='_blank'><strong>Nicu Sebe</strong></a><sup> 6</sup>
</div>

<div align='center'>
    <sup>1 </sup>Nankai University&ensp;  <sup>2 </sup>Northwestern Polytechnical University&ensp;  <sup>3 </sup>National University of Defense Technology&ensp;
    <br />
    <sup>4 </sup>Aalto University&ensp;  <sup>5 </sup>Shanghai AI Laboratory&ensp;  <sup>6 </sup>University of Trento&ensp;
</div>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <a href='https://www.sciopen.com/article/pdf/10.26599/AIR.2024.9150038.pdf'><img src='https://img.shields.io/badge/Journal-Paper-red'></a>&ensp;
  <a href='https://arxiv.org/pdf/2401.03407'><img src='https://img.shields.io/badge/arXiv-Paper-red'></a>&ensp;
  <a href='https://drive.google.com/file/d/1FWvKDWTnK9RsiywfCsIxsnQzqv-dlO5u/view'><img src='https://img.shields.io/badge/中文版-Paper-red'></a>&ensp;
  <a href='https://www.birefnet.top'><img src='https://img.shields.io/badge/Page-Project-red'></a>&ensp;
  <a href='https://drive.google.com/drive/folders/1s2Xe0cjq-2ctnJBR24563yMSCOu4CcxM'><img src='https://img.shields.io/badge/GDrive-Stuff-green'></a>&ensp;
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-yellow'></a>&ensp;
  <a href='https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF-Space-blue'></a>&ensp;
  <a href='https://huggingface.co/ZhengPeng7/BiRefNet'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-blue'></a>&ensp;
</div>

This enhanced version is based on the official implementation of "[**Bilateral Reference for High-Resolution Dichotomous Image Segmentation**](https://arxiv.org/pdf/2401.03407)" (___CAAI AIR 2024___).

|            *DIS-Sample_1*        |             *DIS-Sample_2*        |
| :------------------------------: | :-------------------------------: |
| <img src="https://drive.google.com/thumbnail?id=1ItXaA26iYnE8XQ_GgNLy71MOWePoS2-g&sz=w400" /> |  <img src="https://drive.google.com/thumbnail?id=1Z-esCujQF_uEa_YJjkibc3NUrW4aR_d4&sz=w400" /> |

## 📖 Citation

If you use this enhanced training pipeline in your research, please cite both the original BiRefNet paper and mention this enhanced version:

```bibtex
@article{BiRefNet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
```

And mention: *"We used the enhanced training pipeline from https://github.com/jason-engage/BiRefNet"*

## 🙏 Acknowledgments

- Original BiRefNet implementation by [Peng Zheng](https://github.com/ZhengPeng7) and team
- Fast foreground estimation GPU implementation by [@lucasgblu](https://github.com/lucasgblu)
- All contributors to the original BiRefNet project

## 📄 License

MIT License - same as the original BiRefNet

---

## 🔗 Links

- **Original Repository**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **Enhanced Fork**: [jason-engage/BiRefNet](https://github.com/jason-engage/BiRefNet)
- **Issues**: [Report bugs or request features](https://github.com/jason-engage/BiRefNet/issues)

---

*For model architecture details, dataset information, and inference examples, please refer to the [original repository](https://github.com/ZhengPeng7/BiRefNet).*
# BITM: Boosting Inverse Tone Mapping via Diffusion Regularization

<a href='https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Lu_Boosting_Inverse_Tone_Mapping_via_Diffusion_Regularization_ICCVW_2025_paper.pdf'><img src='https://img.shields.io/badge/Paper-ICCVW2025-b31b1b.svg'></a> &nbsp;&nbsp;

## :trophy: Championship Award of the AIM 2025 Inverse Tone Mapping Challenge

Our team achieved the **Championship Award** in the [AIM 2025 Inverse Tone Mapping Challenge](https://codalab.lisn.upsaclay.fr/competitions/22245), with **34.49 dB PSNR** and **0.95 SSIM** on the final test set.

This is the official PyTorch implementation of the paper:

>**Boosting Inverse Tone Mapping via Diffusion Regularization**<br>
>Xin Lu, Yufeng Peng, Chengjie Ge, Zhijing Sun, Ziang Zhou, Zishun Liao, Zihao Li, Dong Li, Qiyu Kang, Xueyang Fu<sup>&dagger;</sup>, Zheng-Jun Zha<br>
>University of Science and Technology of China (USTC)<br>
>ICCV Workshop 2025

![pipeline](assets/intro.png)


## :wrench: Dependencies and Installation

```bash
git clone https://github.com/xin1u/BITM.git
cd BITM
pip install -r requirements.txt
```

**Main dependencies:** PyTorch >= 1.10, torchvision, numpy, Pillow, timm, tensorboard, lpips


## :file_folder: Project Structure

```
BITM/
    ├── ckpt/                          # Pre-trained checkpoints
    │   ├── best_model.pth             # Final ITM model weights
    │   └── pretrained_model.pth       # Diffusion pre-trained weights
    ├── datasets/                      # Dataset loading
    │   └── datasets_pairs.py
    ├── loss/                          # Loss functions
    │   └── losses.py                  # Charbonnier, FFT, SSIM, LPIPS losses
    ├── networks/                      # Model architectures
    │   ├── bitm_arch.py               # NAFNet U-Net with global residual
    │   ├── diffusion_reg.py           # VP-SDE + L_reg + L_orthog
    │   ├── image_utils.py             # Image splitting & merging
    │   └── local_arch.py              # Local inference wrapper
    ├── utils/
    │   └── UTILS.py                   # Metrics & utilities
    ├── TEST.py                        # Inference script
    ├── train_pretrain.py              # Unconditional diffusion pre-training
    └── train_bitm.py                  # Task regularization fine-tuning (3-stage)
```


## :surfer: Quick Start

**Step 1: Download Checkpoints**

Download the pre-trained checkpoint and place it in the `ckpt/` directory:
- `best_model.pth` — Final inverse tone mapping model

**Step 2: Run Testing**

```bash
python TEST.py \
    --eval_in_path ./test_images/ \
    --result_path ./results/
```

The restored HDR results will be saved in `./results/`. A log file at `./results/log_file/test.txt` records per-image PSNR/SSIM metrics.

**Note:** Ensure both paths end with `/`.


## :muscle: Train

### Phase 1: Unconditional Diffusion Pre-training (Sec.3.1)

Pre-train the restoration network as a VP-SDE denoiser on HDR images:

```bash
python train_pretrain.py \
    --hdr_dir ./data/train_hdr/ \
    --save_path ./ckpt/ \
    --epochs 200 \
    --batch_size 16 \
    --lr 0.0002 \
    --crop_size 256
```

The pre-trained weights are saved as `ckpt/pretrained_model.pth`.


### Phase 2: Task Regularization Fine-tuning (Sec.3.2)

Three-stage progressive training on LDR-HDR pairs:

1. **Stage 1** — Charbonnier + FFT loss (Adam, lr=4e-4, batch=22, patch=256, 1000 epochs):
```bash
python train_bitm.py \
    --experiment_name stage1 \
    --unified_path ./experiments/ \
    --training_path_txt data/train_list.txt \
    --eval_in_path /PATH/val_input/ \
    --eval_gt_path /PATH/val_gt/ \
    --load_pre_model True \
    --pre_model ./ckpt/pretrained_model.pth \
    --BATCH_SIZE 22 \
    --Crop_patches 256 \
    --learning_rate 0.0004 \
    --EPOCH 1000 \
    --base_loss char \
    --addition_loss fft \
    --addition_loss_coff 0.02 \
    --use_diffusion_reg True
```

2. **Stage 2** — Charbonnier + SSIM loss (Adam, lr=4e-5, batch=3, patch=512, 300 epochs):
```bash
python train_bitm.py \
    --experiment_name stage2 \
    --unified_path ./experiments/ \
    --load_pre_model True \
    --pre_model ./experiments/stage1/best_model.pth \
    --BATCH_SIZE 3 \
    --Crop_patches 512 \
    --learning_rate 0.00004 \
    --EPOCH 300 \
    --base_loss char \
    --addition_loss ssim \
    --addition_loss_coff 0.2 \
    --grad_accum_steps 8 \
    --use_diffusion_reg True
```

3. **Stage 3** — Charbonnier + LPIPS loss (SGD, lr=2e-5, batch=1, patch=640, 200 epochs):
```bash
python train_bitm.py \
    --experiment_name stage3 \
    --unified_path ./experiments/ \
    --load_pre_model True \
    --pre_model ./experiments/stage2/best_model.pth \
    --BATCH_SIZE 1 \
    --Crop_patches 640 \
    --learning_rate 0.00002 \
    --EPOCH 200 \
    --base_loss char \
    --addition_loss lpips \
    --addition_loss_coff 0.6 \
    --optim sgd \
    --grad_accum_steps 22 \
    --use_diffusion_reg True
```


## :book: Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Lu_2025_ICCV,
    author    = {Lu, Xin and Peng, Yufeng and Ge, Chengjie and Sun, Zhijing and Zhou, Ziang and Liao, Zishun and Li, Zihao and Li, Dong and Kang, Qiyu and Fu, Xueyang and Zha, Zheng-Jun},
    title     = {Boosting Inverse Tone Mapping via Diffusion Regularization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025}
}
```


## :postbox: Contact

Please feel free to contact us if there is any question (luxion@mail.ustc.edu.cn).

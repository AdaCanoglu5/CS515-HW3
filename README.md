# CS515 HW3 Deep Learning

This project keeps the compact HW2 layout while extending it for the HW3 robustness pipeline.

Core entry points:

- `main.py`: shared training/evaluation entry point
- `train.py`: clean, AugMix, and distillation training
- `test.py`: clean, CIFAR-10-C, PGD, and transfer evaluation
- `parameters.py`: argparse + dataclass config
- `gradcam.py`: offline Grad-CAM generation from saved PGD sample pairs
- `run_experiments.sh`: deterministic end-to-end launcher
- `analysis.ipynb`: artifact-driven analysis notebook

Standard experiment family:

- `teacher_clean_resnet`
- `teacher_augmix_resnet`
- `student_distill_from_clean_teacher`
- `student_distill_from_augmix_teacher`

The teacher runs follow the HW2 `transfer_modify_finetune` procedure:

- pretrained `ResNet-18`
- modified first convolution for CIFAR-10
- removed max-pooling
- full fine-tuning on CIFAR-10

Artifacts are written under `artifacts/`:

- `artifacts/checkpoints/`
- `artifacts/runs/`
- `artifacts/robustness/`
- `artifacts/features/`
- `artifacts/figures/`
- `artifacts/configs/`

Example usage:

```bash
python3 main.py --mode both --dataset cifar10 --model resnet --pretrained true --transfer_mode modify_finetune --train_mode clean_finetune --eval_mode clean --run_name teacher_clean_resnet
python3 main.py --mode test --dataset cifar10 --model resnet --pretrained true --transfer_mode modify_finetune --eval_mode pgd --attack_norm linf --attack_epsilon 4/255 --attack_steps 20 --checkpoint_path artifacts/checkpoints/teacher_clean_resnet.pth --run_name teacher_clean_resnet
python3 gradcam.py --model resnet --dataset cifar10 --pretrained --transfer_mode modify_finetune --checkpoint_path artifacts/checkpoints/teacher_clean_resnet.pth --samples_path artifacts/robustness/teacher_clean_resnet_pgd_linf_samples.pt
```

For the full pipeline, run:

```bash
./run_experiments.sh
```

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_DIR="${DATA_DIR:-./data}"
CIFAR10C_DIR="${CIFAR10C_DIR:-./data/CIFAR-10-C}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"

mkdir -p \
  "${OUTPUT_DIR}/checkpoints" \
  "${OUTPUT_DIR}/runs" \
  "${OUTPUT_DIR}/robustness" \
  "${OUTPUT_DIR}/features" \
  "${OUTPUT_DIR}/figures" \
  "${OUTPUT_DIR}/configs"

python3 main.py \
  --mode both \
  --dataset cifar10 \
  --model resnet \
  --train_mode clean_finetune \
  --eval_mode clean \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name teacher_clean_resnet

python3 main.py \
  --mode both \
  --dataset cifar10 \
  --model resnet \
  --train_mode augmix_finetune \
  --eval_mode clean \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name teacher_augmix_resnet

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model resnet \
  --eval_mode cifar10c \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --cifar10c_dir "${CIFAR10C_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name teacher_clean_resnet \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_clean_resnet.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model resnet \
  --eval_mode cifar10c \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --cifar10c_dir "${CIFAR10C_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name teacher_augmix_resnet \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_augmix_resnet.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model resnet \
  --eval_mode pgd \
  --attack_norm linf \
  --attack_epsilon 4/255 \
  --attack_steps 20 \
  --attack_step_size 1/255 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --export_features true \
  --run_name teacher_clean_resnet \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_clean_resnet.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model resnet \
  --eval_mode pgd \
  --attack_norm l2 \
  --attack_epsilon 0.25 \
  --attack_steps 20 \
  --attack_step_size 0.025 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name teacher_clean_resnet \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_clean_resnet.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model resnet \
  --eval_mode pgd \
  --attack_norm linf \
  --attack_epsilon 4/255 \
  --attack_steps 20 \
  --attack_step_size 1/255 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --export_features true \
  --run_name teacher_augmix_resnet \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_augmix_resnet.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model resnet \
  --eval_mode pgd \
  --attack_norm l2 \
  --attack_epsilon 0.25 \
  --attack_steps 20 \
  --attack_step_size 0.025 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name teacher_augmix_resnet \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_augmix_resnet.pth"

python3 main.py \
  --mode both \
  --dataset cifar10 \
  --model cnn \
  --teacher_model resnet \
  --train_mode distill \
  --eval_mode clean \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --teacher_checkpoint "${OUTPUT_DIR}/checkpoints/teacher_clean_resnet.pth" \
  --run_name student_distill_from_clean_teacher

python3 main.py \
  --mode both \
  --dataset cifar10 \
  --model cnn \
  --teacher_model resnet \
  --train_mode distill \
  --eval_mode clean \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --teacher_checkpoint "${OUTPUT_DIR}/checkpoints/teacher_augmix_resnet.pth" \
  --run_name student_distill_from_augmix_teacher

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --eval_mode cifar10c \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --cifar10c_dir "${CIFAR10C_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name student_distill_from_clean_teacher \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/student_distill_from_clean_teacher.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --eval_mode cifar10c \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --cifar10c_dir "${CIFAR10C_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name student_distill_from_augmix_teacher \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/student_distill_from_augmix_teacher.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --eval_mode pgd \
  --attack_norm linf \
  --attack_epsilon 4/255 \
  --attack_steps 20 \
  --attack_step_size 1/255 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --export_features true \
  --run_name student_distill_from_clean_teacher \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/student_distill_from_clean_teacher.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --eval_mode pgd \
  --attack_norm l2 \
  --attack_epsilon 0.25 \
  --attack_steps 20 \
  --attack_step_size 0.025 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name student_distill_from_clean_teacher \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/student_distill_from_clean_teacher.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --eval_mode pgd \
  --attack_norm linf \
  --attack_epsilon 4/255 \
  --attack_steps 20 \
  --attack_step_size 1/255 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --export_features true \
  --run_name student_distill_from_augmix_teacher \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/student_distill_from_augmix_teacher.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --eval_mode pgd \
  --attack_norm l2 \
  --attack_epsilon 0.25 \
  --attack_steps 20 \
  --attack_step_size 0.025 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name student_distill_from_augmix_teacher \
  --checkpoint_path "${OUTPUT_DIR}/checkpoints/student_distill_from_augmix_teacher.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --target_model cnn \
  --source_model resnet \
  --eval_mode transfer \
  --attack_norm linf \
  --attack_epsilon 4/255 \
  --attack_steps 20 \
  --attack_step_size 1/255 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name transfer_clean_teacher_to_clean_student \
  --source_checkpoint "${OUTPUT_DIR}/checkpoints/teacher_clean_resnet.pth" \
  --target_checkpoint "${OUTPUT_DIR}/checkpoints/student_distill_from_clean_teacher.pth"

python3 main.py \
  --mode test \
  --dataset cifar10 \
  --model cnn \
  --target_model cnn \
  --source_model resnet \
  --eval_mode transfer \
  --attack_norm linf \
  --attack_epsilon 4/255 \
  --attack_steps 20 \
  --attack_step_size 1/255 \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name transfer_augmix_teacher_to_augmix_student \
  --source_checkpoint "${OUTPUT_DIR}/checkpoints/teacher_augmix_resnet.pth" \
  --target_checkpoint "${OUTPUT_DIR}/checkpoints/student_distill_from_augmix_teacher.pth"

if [ -f "${OUTPUT_DIR}/robustness/teacher_clean_resnet_pgd_linf_samples.pt" ]; then
  python3 gradcam.py \
    --model resnet \
    --dataset cifar10 \
    --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_clean_resnet.pth" \
    --samples_path "${OUTPUT_DIR}/robustness/teacher_clean_resnet_pgd_linf_samples.pt" \
    --output_dir "${OUTPUT_DIR}/figures" \
    --device "${DEVICE}" \
    --run_name teacher_clean_resnet
fi

if [ -f "${OUTPUT_DIR}/robustness/teacher_augmix_resnet_pgd_linf_samples.pt" ]; then
  python3 gradcam.py \
    --model resnet \
    --dataset cifar10 \
    --checkpoint_path "${OUTPUT_DIR}/checkpoints/teacher_augmix_resnet.pth" \
    --samples_path "${OUTPUT_DIR}/robustness/teacher_augmix_resnet_pgd_linf_samples.pt" \
    --output_dir "${OUTPUT_DIR}/figures" \
    --device "${DEVICE}" \
    --run_name teacher_augmix_resnet
fi

if python3 -c "import ptflops" >/dev/null 2>&1; then
  python3 main.py --mode flops --dataset cifar10 --model resnet --device "${DEVICE}" --output_dir "${OUTPUT_DIR}" --run_name flops_summary
else
  echo "Skipping FLOPs summary because ptflops is not installed."
fi

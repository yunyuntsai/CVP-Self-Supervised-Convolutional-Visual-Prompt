# declare -a CorruptionArray=("gaussian_noise" "shot_noise" "impulse_noise")

declare -a CorruptionArray=(
                       "brightness" "contrast" "elastic_transform"
                      "pixelate" "jpeg_compression")


declare -a SeverityArray=(1)
declare -a BatchSizeArray=(2 4 8 16 32)

# for c in ${CorruptionArray[@]}
# do
#     for s in ${SeverityArray[@]}
#     do
#         CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 32 --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_119.pth  --output_fn cifar10C_NORM_test_log.csv --eval  --corruption $c  --severity $s --attack_iters 1 --allow_adapt norm --adapt_only
#     done
# done

# for s in ${SeverityArray[@]}
# do
#     for c in ${CorruptionArray[@]}
#     do
#         CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 8 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/ImageNet-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth --output_fn IMNETC_V2_NORM_log.csv  --eval  --corruption $c --severity $s  --attack_iters 1  --allow_adapt norm --adapt_only
#     done
# done



for s in ${SeverityArray[@]}
do
    for c in ${CorruptionArray[@]}
    do
        for bs in  ${BatchSizeArray[@]}
        do  
           CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch $bs --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_119.pth  --output_fn cifar10C_compare_batch_norm_log.csv --eval  --corruption $c  --severity $s --attack_iters 1 --norm l_2 --allow_adapt norm --adapt_only
        done
    done
done
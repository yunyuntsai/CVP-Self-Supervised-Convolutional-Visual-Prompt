# declare -a CorruptionArray=("gaussian_noise" "shot_noise" "impulse_noise"
#                         "defocus_blur" "motion_blur" "glass_blur" "zoom_blur" "snow" 
#                       "frost" "fog" "brightness" "contrast" "elastic_transform"
#                       "pixelate" "jpeg_compression")
declare -a CorruptionArray=("impulse_noise"
                        "defocus_blur" "motion_blur" "glass_blur" "zoom_blur" "snow" 
                      "frost" "fog" "brightness" "contrast" "elastic_transform"
                      "pixelate" "jpeg_compression")

declare -a SeverityArray=(1 2 3 4 5)


for c in ${CorruptionArray[@]}
do
    for s in ${SeverityArray[@]}
    do
        CUDA_VISIBLE_DEVICES=4 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 64 --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_119.pth  --output_fn cifar10C_TENT_test_log.csv --eval  --corruption $c  --severity $s --attack_iters 1 --allow_adapt tent --tent_only
    done
done
declare -a CorruptionArray=("gaussian_noise"  "shot_noise" "impulse_noise"
                        "defocus_blur" "motion_blur" "glass_blur" "zoom_blur" "snow" 
                      "frost" "fog" "brightness" "contrast" "elastic_transform"
                      "pixelate" "jpeg_compression")


declare -a SeverityArray=(2 3 4 5)


# for c in ${CorruptionArray[@]}
# do
#     for s in ${SeverityArray[@]}
#     do
#         CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 32 --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_119.pth  --output_fn cifar10C_NORM_test_log.csv --eval  --corruption $c  --severity $s --attack_iters 1 --allow_adapt norm --adapt_only
#     done
# done

for s in ${SeverityArray[@]}
do
    for c in ${CorruptionArray[@]}
    do
        CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 8 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/ImageNet-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth --output_fn IMNETC_V2_NORM_log.csv  --eval  --corruption $c --severity $s  --attack_iters 1  --allow_adapt norm --adapt_only
    done
done

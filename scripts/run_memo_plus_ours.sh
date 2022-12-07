declare -a CorruptionArray=("gaussian_noise" "shot_noise" "impulse_noise"
                        "glass_blur" "defocus_blur")
# declare -a CorruptionArray=( "zoom_blur" "motion_blur" "brightness" "snow" 
                    #   "frost" "fog"  "contrast"  "pixelate" "jpeg_compression" "elastic_transform")

declare -a SeverityArray=(1)

# for c in ${CorruptionArray[@]}
# do
#     for s in ${SeverityArray[@]}
#     do
#         CUDA_VISIBLE_DEVICES=5 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 16 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ImageNetC-customize --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth --output_fn IMNETC_V2_MEMO_plus_OURS_log.csv  --eval  --corruption $c --severity $s  --attack_iters 1 --aug_name sharpness --allow_adapt memo 
#     done
# done


for c in ${CorruptionArray[@]}
do
    for s in ${SeverityArray[@]}
    do
       CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 16 --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_199.pth  --output_fn cifar10c_V2_MEMO_plus_OURS_log.csv --eval  --corruption $c --severity $s  --attack_iters 1  --aug_name sharpness --allow_adapt memo
    done
done





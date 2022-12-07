
# declare -a CorruptionArray=("frost")


declare -a CorruptionArray=("gaussian_noise" "shot_noise" "impulse_noise"
                        "glass_blur" "defocus_blur" "zoom_blur" "motion_blur" "brightness"  "snow" 
                      "frost" "fog"  "contrast" "pixelate" "jpeg_compression" "elastic_transform")

declare -a SeverityArray=(4 5)

declare -a AttackItersArray=(1)

declare -a BatchSizeArray=(2 4 8 16)

declare -a EpochArray=(0 9 19 39 59 79 90 119)






# for c in ${CorruptionArray[@]}
# do
#     for s in ${SeverityArray[@]}
#     do
#         CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 16 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ImageNetC-customize --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth  --output_fn IMNETC_OURS_test_log.csv  --eval  --corruption $c --severity $s  --attack_iters 1 --aug_name sharpness 
#     done
# done


# for s in ${SeverityArray[@]}
# do
#     for  c in ${CorruptionArray[@]}
#     do
#         CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 16 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/ImageNet-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth --output_fn IMNETC_V2_OURS_log.csv  --eval  --corruption $c --severity $s  --attack_iters 5 --aug_name sharpness
#     done
# done

# CUDA_VISIBLE_DEVICES=3 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 16 --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_199.pth  --output_fn cifar10c_V2_L2_log.csv  --eval  --corruption $c --severity $s  --attack_iters $ai  --norm l_2

# for  c in ${CorruptionArray[@]} 
# do
#     for s in ${SeverityArray[@]}
#     do
#         CUDA_VISIBLE_DEVICES=3 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 8 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/ImageNet-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth --output_fn IMNETC_V2_OURS_log.csv  --eval  --corruption $c --severity $s  --attack_iters 1 --aug_name sharpness --allow_gcam
#     done
# done
#  CUDA_VISIBLE_DEVICES=3 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch $bs --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/ImageNet-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth --output_fn IMNETC_compare_batch_ours_log.csv  --eval  --corruption $c --severity $s  --attack_iters 1 --aug_name sharpness 
            # CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch $bs --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_199.pth  --output_fn cifar10c_compare_batch_ours.csv  --eval  --corruption $c --severity $s  --attack_iters 1  --norm l_2 --aug_name sharpness



for s in ${SeverityArray[@]}
do
    for c in ${CorruptionArray[@]}
    do
        CUDA_VISIBLE_DEVICES=2 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 16 --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_199.pth  --output_fn cifar10c_V2_sharp_random_comp_kernel_s4s5.csv  --eval  --corruption $c --severity $s  --attack_iters 5  --norm l_2 --aug_name sharpness --update_kernel
        # CUDA_VISIBLE_DEVICES=7 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 8 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/ImageNet-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth --output_fn IMNETC_V2_OURS_kernel_random.csv  --eval  --corruption $c --severity $s  --attack_iters 5 --aug_name sharpness --update_kernel
    done
done



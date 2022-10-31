declare -a SeverityArray=(2 3 4 5)


# for c in ${CorruptionArray[@]}
# do
for s in ${SeverityArray[@]}
do
    CUDA_VISIBLE_DEVICES=1 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 16 --data-dir ../cifar-data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/CIFAR-10-C/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/cifar10_standard.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/cifar10c_4/ssl_contrast_199.pth  --eval  --corruption all --severity $s  --attack_iters 5 --aug_name sharpness 
done
# done

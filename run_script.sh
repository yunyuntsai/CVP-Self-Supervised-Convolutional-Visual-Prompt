declare -a SeverityArray=(2 3 4 5)


# for c in ${CorruptionArray[@]}
# do
for s in ${SeverityArray[@]}
do
    CUDA_VISIBLE_DEVICES=1 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 16 --data-dir ../ImageNet-Data/ --corr-dir /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ImageNetC-customize/ --md_path /local/rcs/yunyun/SelfSupDefense-random-experiments/resnet50.pth  --ckpt /local/rcs/yunyun/SelfSupDefense-random-experiments/data/ckpts/imagenetc_2/ssl_contrast_best.pth  --eval  --corruption all --severity $s  --attack_iters 1 --allow_adapt
done
# done

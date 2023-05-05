


declare -a CorruptionArray=( "gaussian_noise", "impulse_noise", "shot_noise", 
                     "defocus_blur" "zoom_blur" "motion_blur" "brightness"  "snow" 
                      "frost" "fog"  "contrast" "pixelate" "jpeg_compression" "elastic_transform")


declare -a SeverityArray=(1 2 3 4 5)

declare -a AttackItersArray=(5)

declare -a BatchSizeArray=(16)




for s in ${SeverityArray[@]}
    do
        for c in ${CorruptionArray[@]}
        do
            for a in ${AttackItersArray[@]}
            do
            CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 16 --data-dir ../cifar-data/ --corr-dir-[Path of CIFAR-10-C] --md_path [Path of backbone.pth]  --ckpt [Path of SSL model.pth]  --output_fn log.csv  --eval  --corruption $c --severity $s  --attack_iters $a  --norm 'l_2' --aug_name 'conv' --update_kernel 'rand3'
        done
    done
done




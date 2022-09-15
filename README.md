# Self-Supervision-Test-Time-Adaptation

- Training Contrastive SSL model for Cifar10-C

```
CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 16 \\
--data-dir ../cifar-data --md_path ./ckpt/cifar10_standard.pth --output_dir output/train_cifar10c_ssl --epoch 100 \\
```

- To reverse Cifar10-C at Testing phase, run following line

```
CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_cifar10c.py --eval --batch-size 32 --test_batch 16 \\
--data-dir ../cifar-data/ --corr-dir ./data/CIFAR-10-C-customize/ 
--md_path  ./ckpt/cifar10_standard.pth  \\
--ckpt  ./ckpt/cifar10c_ssl/ssl_contrast_199.pth \\ 
--output_dir output/{test_log save path} \\
--corruption {corruption type}  --severity {severity level 1 --> 5} --aug_name sharpness --attack_iters 1 
```
 
 - 15 corruption types:  
 > gaussian_noise, shot_noise, impulse_noise,  
 > defocus_blur, motion_blur, glass_blur, zoom_blur,  
 > snow, frost, fog, brightness,  
 > contrast,elastic_transform, pixelate, jpeg_compression


- Training Contrastive SSL model for ImageNet-C
```
 CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 16 \\
 --data-dir ../ImageNet-Data/ --md_path ./ckpt/resnet50.pth  --output_dir output/train_imgnetc_ssl --epoch 100 \\
```

- To reverse ImageNet-C at Testing phase, run following line
```
CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_imgnetc.py --batch-size 32 --test_batch 16  --eval \\
--data-dir ../ImageNet-Data/ --corr-dir ./data/ImageNetC-customize/ --md_path ./resnet50.pth  --ckpt ./data/ckpts/imagenetc_ssl/ssl_contrast_best.pth  
--corruption {corruption type}  --severity {severity level 1 --> 5} --aug_name sharpness --attack_iters 1 --output_dir output/{test_log save path}
```

- For the pretrained checkpoint, please download from following links
> ResNet50 checkpoint: [a link](https://drive.google.com/file/d/1tDW8-HCltiI_ECQgRDb-piXHweZdFt9B/view?usp=sharing)
> 

# Self-Supervision-Test-Time-Adaptation

- Training Contrastive SSL model for Cifar10-C

```
CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_cifar10c.py --batch-size 32 --test_batch 16 \\
--data-dir {original cifar10 data path} --md_path ./ckpt/cifar10_standard.pth --output_dir output/train_cifar10c_ssl
```

- To reverse Cifar10-C at Testing phase, run following cmd line

```
CUDA_VISIBLE_DEVICES=0 python3 ssl_reversed_cifar10c.py --eval --batch-size 32 --test_batch 16 \\
--data-dir ../cifar-data/ --corr-dir ./data/CIFAR-10-C-customize/ 
--md_path  ./ckpt/cifar10_standard.pth  \\
--ckpt  ./ckpt/cifar10c_4/ssl_contrast_199.pth \\ 
--output_dir output/test_gaussian \\
--corruption {corruption type}  --severity {severity level} --aug_name sharpness --attack_iters 1 
```
 
 > 15 corruption types:  
 >> gaussian_noise, shot_noise, impulse_noise,  
 >> defocus_blur, motion_blur, glass_blur, zoom_blur,  
 >> snow, frost, fog, brightness,  
 >> contrast,elastic_transform, pixelate, jpeg_compression

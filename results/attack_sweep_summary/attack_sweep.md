# Attack generalization of calibrated DP-SGD (eps=5)

Each cell is `Attack F1 / LPIPS / PSNR`. Clean = no defense; DP-SGD = calibrated DP-SGD with eps=5, delta=1e-5, C=1.0.

## medical (pneumonia_descriptive)

| Attack | Clean F1 | Clean LPIPS | Clean PSNR | DP-SGD F1 | DP-SGD LPIPS | DP-SGD PSNR |
|---|---|---|---|---|---|---|
| DLG | 0.375 |  1.370 |  8.246 | 0.375 |  1.406 |  8.278 |
| IG | 0.750 |  0.868 | 10.131 | 0.375 |  0.787 | 11.345 |
| GradInversion | 0.750 |  0.886 |  8.580 | 0.500 |  0.864 |  8.476 |
| HF-GradInv | 0.750 |  1.007 | 11.732 | 0.125 |  0.791 | 11.412 |

## uav (solar_panels)

| Attack | Clean F1 | Clean LPIPS | Clean PSNR | DP-SGD F1 | DP-SGD LPIPS | DP-SGD PSNR |
|---|---|---|---|---|---|---|
| DLG | 0.000 |  1.223 |  8.223 | 0.000 |  1.176 |  9.436 |
| IG | 0.250 |  0.686 | 12.024 | 0.000 |  0.789 | 12.471 |
| GradInversion | 0.250 |  0.701 | 10.411 | 0.000 |  0.718 |  8.923 |
| HF-GradInv | 0.125 |  0.655 | 12.369 | 0.000 |  0.747 | 12.551 |


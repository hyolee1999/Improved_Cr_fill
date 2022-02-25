# Improved_Cr_fill
In this project ,I intend to optimize deep learning inpainting model that can run on mobile device so I try to improve the "CR-Fill: Generative Image Inpainting with Auxiliary Contextual Reconstruction" by replacing gated convolution with **light weight gated convolution** , initialize weights with the methods proposed in paper [Checkerboard artifact free sub-pixel convolution](https://arxiv.org/abs/1707.02937) as I replace all Upsampling layers with sub-pixel convolutions.
I use in total two models:
- On low resolution ( below 512 x 512) : I use improved CR-Fill architecture on low resolution.
- On high resolution ( above 512x512) : I reused pretrained CR-Fill architecture on high resolution .I also add another module so-called
Contextual Residual Aggregation 
The detail of **light weight gated convolution** and **Contextual Residual Aggregation** can be found in [Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting](https://arxiv.org/abs/2005.09704).

## Train

1. If you want to train from scratch , you need to remove checkpoints, create flist.py of paths to images used to train which will be more detail in [edge-connect ](https://github.com/knazeri/edge-connect) and create config.py of hyper-parameters as in generative_inpaintingv2/contextual_loss_fix_mask_wise_and_pixel_shuffle/checkpoint/checkpoint_change_dis or reuse config.py  if you can to continue the training

2. You can also check my [flist.py](https://drive.google.com/drive/folders/1IuoeYJKbhN0M-RnhDhzfwX2JqOcqILm2?usp=sharing) if you want

3. The file for training is **contextual_loss_fix_mask_wise_and_pixel_shuffle_change_dis.ipynb** , you can check out it and change the directory based on yours 
## Validation
- You also create flist.py  of paths to images used to validate and write it to same config.py for training or reuse my [flist.py](https://drive.google.com/drive/folders/1IuoeYJKbhN0M-RnhDhzfwX2JqOcqILm2?usp=sharing)

## Testing and Evaluating
- For testing and evaluating , you can checkout the **validation_fix.ipynb** for more detail






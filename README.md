# CMU-02699-Image-Segmentation-via-GANs2
Spring 2022 Bioimage Informatics (Self-Study) project

This project explores using a deep generative adversarial network (GAN) to perform semi-supervised image segmentation on the 2015 MICCAI Gland Challenge dataset. This is an improvement over version 1, adding triplet loss and hard negative mining to the training process. 

See version 1 [here](https://github.com/ryanquinnnelson/CMU-02699-Image-Segmentation-via-GANs).

See the final report [here](https://github.com/ryanquinnnelson/CMU-02699-Image-Segmentation-via-GANs-v2/blob/main/docs/Nelson_Parameterized_GAN_architecture.pdf).


## How to run on an EC2 instance
1. Copy `octopus` library to Deep Learning instance. 
2. Copy data to instance. Data was is found at [here](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/). I organized the files into the following format.
```
warwick
    ├── a
    ├── a_anno
    ├── b
    ├── b_anno
    ├── training
    └── training_anno
```
I also padded all the numbering in the filenames (i.e. testA_1_anno.bmp -> testA_01_anno.bmp)

4. Copy this codebase to instance.
5. Log into instance.
6. (Optional) Mount drive on EC2.
```
bash octopus/bin/mount_drive
```
6. Activate `conda` environment.
```
conda activate pytorch_p38
```
7. Install `wandb`.
```
bash octopus/bin/setup_wandb
```
8. Run code for a single run or a sweep.

Single run
```
python /home/ubuntu/CMU-02699-Image-Segmentation-via-GANs2/run_octopus.py --filename=/home/ubuntu/CMU-02699-Image-Segmentation-via-GANs2/configs/remote/config_remote_GAN_only_002.txt
```
Sweep
```
wandb sweep /home/ubuntu/CMU-02699-Image-Segmentation-via-GANs2/sweeps/remote/sweep_remote_grid_023.yaml
````







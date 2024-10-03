## RobustMedSeg &mdash; Official PyTorch Implementation
<div align="justify">

![Teaser image](./fed_grid.png)
**Picture:** *Comparative analysis of the segmentation results: heart segmentation w/ scanner Siemens (top), vessel segmentation w/ OCTA imagery (center), and brain segmentation w/ PET scans (bottom) using different methods.*

This repository allows users to produce accurate segmentation with minimal training and annotation efforts. It contains the official PyTorch implementation of the following paper:

> **Federated Multi-Centric Image Segmentation with Uneven Label Distribution**<br>
> Francesco Galati, Rosa Cortese, Ferran Prados, Marco Lorenzi, Maria A. Zuluaga<br>
> In: Medical Image Computing and Computer Assisted Intervention – MICCAI (2024)
>
> **Abstract:** *While federated learning is the state-of-the-art methodology for collaborative learning, its adoption for training segmentation models often relies on the assumption of uniform label distributions across participants, and is generally sensitive to the large variability of multi-centric imaging data. To overcome these issues, we propose a novel federated image segmentation approach adapted to complex non-iid setting typical of real-life conditions. We assume that labeled dataset is not available to all clients, and that clients data exhibit differences in distribution due to three factors: different scanners, imaging modalities and imaged organs. Our proposed framework collaboratively builds a multimodal data factory that embeds a shared, disentangled latent representation across participants. In a second asynchronous stage, this setup enables local domain adaptation without exchanging raw data or annotations, facilitating target segmentation. We evaluate our method across three distinct scenarios, including multi-scanner cardiac magnetic resonance segmentation, multi-modality skull stripping, and multi-organ vascular segmentation. The results obtained demonstrate the quality and robustness of our approach as compared to the state-of-the-art methods.*

</div>

## System requirements
- batchgenerators==0.25
- evalutils==0.4.2
- matplotlib==3.8.2
- MedPy==0.4.0
- nibabel==5.2.0
- nilearn==0.10.3
- opencv-python==4.6.0.66
- open-clip-torch==2.24.0
- pytorch-msssim==1.0.0
- scikit-image==0.22.0
- SimpleITK==2.3.1
- tensorboard==2.15.1
- torch==1.13.1+cu116

## Preparing datasets for training

Please refer to the homonymous section in `MultiMedSeg/README.md`.

## Training networks

### Phase 1

This phase requires one additional step to prepare the training data, which each client must execute by running the following command:

```
python prepare_data.py ${K1_dir}/train/ --out ${D1_dir} --size 512 --label 0
...
python prepare_data.py ${Kk_dir}/train/ --out ${Dk_dir} --size 512 --label K-1
```

At this stage, you are ready to begin the federated training of the multimodal data factory. To configure the multi-centric environment, please consult the [fedbiomed documentation](https://fedbiomed.org/latest/tutorials/installation/1-setting-up-environment/). Once all nodes are operational, the researcher can initiate the training by running:

```
python train.py --size 512 --n_sample 3 --n_domains 3 --augment --num_updates_per_round 2000 --rounds 350 --clip-weight 0
```

To enhance convergence and smoother integration across different domains, we run a refinement stage of 35 rounds with 200 iterations each:

```
python train.py --size 512 --n_sample 3 --n_domains 3 --augment --num_updates_per_round 200 --rounds 385 --clip-weight 50
```

After the training is completed, you can aggregate the weights from all nodes by using the following command:

```
python save_global_model.py
```

### Phase 2

<div align="justify">

Please refer to the homonymous section in `MultiMedSeg/README.md`. However, note that you may not have permission to use source data during domain adaptation due to privacy constraints. To circumvent this limitation, you can use the data factory to generate synthetic labeled source data. To do so, run the script `generate_volumes.py` included in this repository:

</div>

```
python generate_volumes.py --ckpt_dir=${SRC_exp_dir} --label=0 --n_volumes=20 --slices_in_one_z=10 --n_segmentation_labels=3 --out_dir=${SRC_fake_dir}
```

## Running inference

Please refer to the homonymous section in `MultiMedSeg/README.md`.

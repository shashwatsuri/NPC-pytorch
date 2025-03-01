# NPC: Neural Point Characters from Video

### [Paper](https://arxiv.org/abs/2304.02013) | [Supplementary](https://lemonatsu.github.io/files/npc/supp.pdf) | [Website](https://lemonatsu.github.io/npc/) 
![](imgs/front.png)
>**NPC: Neural Point Characters from Video**\
>[Shih-Yang Su](https://lemonatsu.github.io/), [Timur Bagautdinov](https://scholar.google.ch/citations?user=oLi7xJ0AAAAJ&hl=en), and [Helge Rhodin](http://helge.rhodin.de/)\
>ICCV 2023

### Updates
- Sep 4, 2023: Code updated with some bug fixes.
- Sep 2, 2023: Fixed a major bug, and re-uploaded the re-extracted point clouds [here](https://drive.google.com/drive/folders/1tdTQDgu0lvJWxMu-xOOLxg-ilVos0EB9?usp=sharing).

This repo also supports [DANBO](https://github.com/LemonATsu/DANBO-pytorch) training. 
For ease of comparisons, we provide the [rendering results](https://drive.google.com/file/d/18dpTxbcCi28M_vHduSJxi5TfpBoyUa8Q/view?usp=sharing) for NPC, DANBO, and TAVA on H3.6M. 

## Setup
```
conda create -n npc python=3.9
conda activate npc

# install pytorch for your corresponding CUDA environments
pip install torch

# install pytorch3d: note that doing `pip install pytorch3d` directly may install an older version with bugs.
# be sure that you specify the version that matches your CUDA environment. See: https://github.com/facebookresearch/pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu114_pyt1110/download.html

# install other dependencies
pip install -r requirements.txt

```
## Training
You can find relevant training configuration in `configs/`.
### Point cloud extraction
To train NPC, we need to first extract surface point clouds. 

You can download pre-extracted point clouds [here](https://drive.google.com/drive/folders/1tdTQDgu0lvJWxMu-xOOLxg-ilVos0EB9?usp=sharing) or use our [example script](https://github.com/LemonATsu/NPC-pytorch/blob/main/point_extraction.sh) for extracting point clouds with DANBO by running:
```
./point_extraction.sh # extract point clouds with DANBO
```

### NPC training
Then, you can train NPC with the following command:
```
python train.py --config-name npc --basedir logs  expname=NPC_h36m_S9 dataset.subject=S9
```
The `config-name npc` corresponds to config file `configs/npc.yaml`, and `dataset.subject=S9` overwrite dataset related configuration, which can be found in `configs/dataset/`. The training logs and checkpoints will be saved in `logs/NPC_h36m_S9/`.

Note that you can change the paths to the point clouds in the config (e.g., [here](https://github.com/LemonATsu/NPC-pytorch/blob/main/configs/npc.yaml#L15)).

## Testing
You can use [`run_render.py`](run_render.py) to render the learned models under different camera motions, or retarget the character to different poses by
```
python run_render.py --config-name h36m_zju model_config=logs/NPC_h36m_S9/config.yaml +ckpt_path=[path/to/specific/ckpt] output_path=[path/to/output] render_dataset.bkgd_to_use=black
```
Here, we render the dataset as specified in config file `configs/render/h36m_zju.yaml` with the model configuration and weights we saved before, and store the output in `output_path`.
	
Config files related to rendering or testing the models are located at `configs/render`. For example, we have `configs/render/h36m_zju_mesh.yaml`for extracting meshes with Marching cubes.

## Dataset
You can find dataset configuration in `configs/dataset`.

We are not allowed to share the pre-processed data for H3.6M and MonoPerfcap due to license terms. If you need access to the pre-trained models and the pre-processed dataset, please reach out to `shihyang[at]cs.ubc.ca`.

## TODOs:
- [x] Add closest box search as described in the paper
- [ ] Add support for box-free point clouds


## Citation
```
@inproceedings{su2023iccv,
    title={NPC: Neural Point Characters from Video},
    author={Su, Shih-Yang and Bagautdinov, Timur and Rhodin, Helge},
    booktitle={International Conference on Computer Vision},
    year={2023}
}
```
```
@inproceedings{su2022danbo,
    title={DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks},
    author={Su, Shih-Yang and Bagautdinov, Timur and Rhodin, Helge},
    booktitle={European Conference on Computer Vision},
    year={2022}
}
```

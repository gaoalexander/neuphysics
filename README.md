# NeuPhysics

#### Data Convention
The data is organized as follows:

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
|-- sparse_points_interest.ply    # contains scene ROI
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

Here the `cameras_xxx.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Setup

Clone this repository

```shell
git clone https://github.com/YilingQiao/NeuPhysics.git
cd NeuPhysics
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>

  - torch==1.8.0
  - opencv_python==4.5.2.52
  - trimesh==3.9.8 
  - numpy==1.19.2
  - pyhocon==0.3.57
  - icecream==2.1.0
  - tqdm==4.50.2
  - scipy==1.7.0
  - PyMCubes==0.1.2

</details>

### Running

- **Training without mask**

```shell
python run.py --mode train --conf ./confs/neuphysics_default.conf --case <case_name>
```

- **Extract surface from trained model** 

```shell
python run.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

- **View interpolation**

```shell
python run.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding image set of view interpolation can be found in `exp/<case_name>/<exp_name>/render/`.

### Train NeuPhysics with your own data

More information can be found in [preprocess_custom_data](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data).

## Citation

If you find this repository helpful, please consider citing our paper as well as others that our work built upon:

```
@article{qiaoandgao2022neuphysics,
  title={NeuPhysics: Editable Neural Geometry and Physics from Monocular Videos},
  author={Qiao, Yi-Ling and Gao, Alexander and Lin, Ming},
  journal={ #TODO },
  year={2022}
}
@article{wang2021neus,
  title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
  author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2106.10689},
  year={2021}
}
@misc{tretschk2020nonrigid,
      title={Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene From Monocular Video},
      author={Edgar Tretschk and Ayush Tewari and Vladislav Golyanik and Michael Zollhöfer and Christoph Lassner and Christian Theobalt},
      year={2020},
      eprint={2012.12247},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

Our code borrows from [NeuS](https://github.com/Totoro97/NeuS) and [NonRigid NeRF](https://github.com/facebookresearch/nonrigid_nerf/).
Additionally, some code snippets are borrowed from [IDR](https://github.com/lioryariv/idr) and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).

Thank you to the authors of these projects for their great work.

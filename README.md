# Coupled Laplacian Eigenmaps for Locally-Aware 3D Rigid Point Cloud Matching
### Citation

Our paper has been accepted at CVPR 2024 and is available on [ArXiv](https://arxiv.org/abs/2402.17372). Please cite our work with
```sh
  
@misc{bastico_coupled_2024,
	title = {Coupled {Laplacian} {Eigenmaps} for {Locally}-{Aware} {3D} {Rigid} {Point} {Cloud} {Matching}},
	url = {http://arxiv.org/abs/2402.17372},
	author = {Bastico, Matteo and Decencière, Etienne and Corté, Laurent and Tillier, Yannick and Ryckelynck, David},
	month = feb,
	year = {2024}
  }
  ```

# Dataset Download

The dataset used in our experiments can be downloaded at the following links, upon request, if needed:

Anomaly Detection:
- MVTEC 3D-AD https://www.mvtec.com/company/research/datasets/mvtec-3d-ad 
- The evaluation code to compute AUC-PRO on the test, as provided in the paper, is also available at the same link

Bone Side Estimation:
- Imperial College London (ICL): https://zenodo.org/records/167808, extract the femur meshes in `data/ICL/Femur` and
tibia meshes in `data/ICL/Tibia`.
- SSM-Tibia: https://simtk.org/projects/ssm_tibia, extract data in the `data/SSM-Tibia` folder.
- Fischer et al. : https://zenodo.org/records/4280899, extract data in the `data/Fischer` folder.

# Requirements 
This code has been tested on Ubuntu 22.04 and mac0S 12.6.7 with python 3.9. 

**Python Note**: depending on the system, `python` in the following commands might be replaced with `python3`. 

- Create and lunch your conda environment with
```
conda create -n CoupledLaplacian
conda activate CoupledLaplacian
```
  
- Install dependencies
```
pip install -r requirements.txt
```

# Data Preparation

Anomaly Detection: The MVTec 3D-AD dataset already comes in the propoer format to be used with our code. Therefore, it does not need any pre-processing. Unzip it in the `data/` folder.

Bone Side Estimation: the pre-processing script for the BSE datasets is `data/prepare_bse.py` and can be run in the following way

```
python data/prepare_bse.py --dataset ICL --dir data/ICL --save_dir data/BSE_Human
```

Here, the options for dataset are ICL, SSM-Tibia and Fischer. The bone point clouds will be saved in the `data/BSE_Human` folder with the following structure

```
BSE_Human
  ├── Femur_L	# Left femur folder
  │   ├── sample_id001.pkl
  │   ├── sample_id002.pkl
  │   ...
  ├── Femur_R
  ├── Tibia_L
  ├── Tibia_R
  ├── Hip_L
  ├── Hip_R
```

Please, if you pre-process the dataset, very that this is the final folder structure you obtain.

# Prediction

For both tasks we provide two ways of perform prediction:

- Single sample inference using Coupled Laplacian with a script in which the algorithm is unfolded, so easy to read, and with graphical feedbacks
- Easy benchmarking on the whole dataset through the model classes we provide (see model package)

## BSE

Single inference of spectral-based BSE can be done as in the following example:

```
python bse.py \
    --source 'data/BSE_Human/Femur_L/C25LFE.pkl' \
    --target 'data/BSE_Human/Femur_R/C04RFE.pkl' \
    --cross 0.5 \
    --voxel_size 2 \
    --maps 20
```

Running this script you should see various plots representing the 3D registration results of both source and mirrored source, the coupled adjacency matrix plot and the first 5 coupled eigenmaps (note that you have to close the plt plot in order to contiue running the script). Finally, you should have the final prediction such as `RESULT: Bones are from OPPOSITE sides`


Benchmarking on the whole datasets for a bone class, as explained in the manuscript, can be performedas 

```
python benchmark_bse.py \
    --dir_left 'data/BSE_Human/Femur_L' \
    --dir_right 'data/BSE_Human/Femur_R/' \
    --method spectral \
    --threshold 13000 \
    --cross 0.5 \
    --voxel_size 2 \
    --maps 20 \
    --output_dir 'results/BSE_Femur'
```

More details on the arguments can be found using `python benchmarks/bse.py --help`. Possible options for `--method` are 'spectral', 'chamfer', 'hausdorff', 'FPFH'. The benchmarking results are saved in the specified folder in a dict (.pkl file) with the following structure in order to be further analysed.

```
{'sample_id001': {'sample_id002': 0/1, ...}, ...}
```

## Anomaly Detection

Single inference of anomaly localization with Coupled Laplacian can be done as in the following example:

```
python ad.py \
    --source 'data/MVTec 3D-AD/bagel/train/good/xyz/000.tiff' \
    --target 'data/MVTec 3D-AD/bagel/test/crack/xyz/005.tiff' \
    --cross 0.5 \
    --voxel_size 0.001 \
    --threshold 0.001 \
    --points 13000 \
    --cpd_down 3000 \
    --maps 200
```

Running this script a browser window should open to visualize the interactive 3D object with the anomaly localization obtained through point-wise comparison of aligned spectral embeddings.

Benchmarking on the whole datasets for an object class can be performed as 

```
python benchmark_ad.py \
    --cls bagel \
    --source 'data/MVTec 3D-AD/bagel/train/good/xyz/000.tiff' \
    --dataset_dir 'data/MVTec 3D-AD' \
    --output_dir 'results/MVTec_3D' \
    --method spectral \
    --cross 0.5 \
    --voxel_size 0.001 \
    --threshold 0.001 \
    --points 13000 \
    --cpd_down 3000 \
    --maps 200
```

More details on the arguments can be found using `python benchmarks/anomaly_detection.py  --help`. Possible options for `--method` are 'spectral', 'Euclidean', 'GPS', 'Hist'.
The benchmarking results are saved in .tiff format in the specified output folder. They can be directly evaluated using the evaluation scripts provided at https://www.mvtec.com/company/research/datasets/mvtec-3d-ad by the datasets authors.

**Result Note**: scirpts are seeded as we did in our experiments, unfortunately randomness may still be present using different machines and therefore the final results may slightly differ from the ones reported in the manuscript. 

# HUTS

This repository comprises of the code base used in the following HUTS [ Hierarchical Unsupervised Topological SLAM ] [paper](https://arxiv.org/abs/2310.04802). 

- Habitat dataset generation script `src/save_habitat.py` can be found on this link: [Link](https://github.com/dasupradyumna/Global-Descriptor). However, for that you need to install Habitat too, that is also explained on `Readme.md` of repo itself. One can use the following link to download Matterport3D files [MatterPort3D](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ayush_sharma_students_iiit_ac_in/EbYF5L1xyctJp7rDvWTsg8QBZAk0ZoqQyHQBttWTRdomvQ?e=5UGrms)

**Currently, this repo comprises of following:**
- Global Retrieval algorithms [GR folder]
    - NetVLAD
    - DIR(GEM)
- HUTS loop pair detection scripts [SCRIPTS folder]
- Local Feature matching algorithms [LFM folder]
    - RoRD
- Backend Optimisation code [BO folder]


**NOTE 1**
For sequence descriptor extractor, we have utilised following repository [link](https://github.com/vandal-vpr/vg-transformers). Follow section `2B and 4A` from the [paper](https://arxiv.org/abs/2310.04802) to know how to utilise seqVLAD model for this use case.

**NOTE 2**
We recommend `conda` environment with `python3.8` on `linux OS`. Use `requirements.txt` for getting required dependencies to run this codebase. You'll have to separately install `G2O cli from source`. 


### Dataset

This code assumes you to have a `original_data` folder having following tree structure:
```txt
original_data
| - color
    | - 1.jpg
    | - 2.jpg
    .
    .
    .
    | - 100.jpg
| - depth
    | - 1.png
    | - 2.png
    .
    .
    .
    | - 100.png
| - poses.csv[x, y, z, qw, qx, qy, qz]
```

**Note 3** 
- We have tested full pipeline on SE2 sequence for Habitat dataset. 

- One can find such SE2 sample dataset over this link: 
    - `original_folder` - [Link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ayush_sharma_students_iiit_ac_in/EhQ7zmOkyHxBuBkNt-yrcrwBPtfuqMMx04LYdZ4_uts7kw?e=iv7d2Q)
    - `data folder` - [Link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ayush_sharma_students_iiit_ac_in/Emt53gYzG4FCrVC8qDl6sVUBm17zmiVmUFaoLNr92i2kqg?e=zykZqY)


## Global Retrieval



### NetVLAD



#### Dataformatting

- Break the data into two seqeuence first part will be considered as reference set and later part as query set. For example breakpoint = 50 for 100 images/poses. Depends on user!

```python
python GR/NetVLAD/DataFormat/break_into_two.py breakpoint path_to_original_data

example: python GR/NetVLAD/DataFormat/break_into_two.py 600 ../original_data/

```

- Make a new folder `data` for keeping NetVLAD processing files and processed files. We'll keep 640*480 size reference & query image in new place, as NetVLAD work best with those dimentsions. We will chose few of the reference and query say every 3rd or 4th images for NetVLAD run.:

```python
python GR/NetVLAD/DataFormat/custom_format.py FREQ "path_to_original_data/reference"  "path_to_data_folder" "name_folder_for_keeping_references_data"

example: python GR/NetVLAD/DataFormat/custom_format.py 3 ../original_data/reference/ ../data/ Reference_resized

```

```python
python GR/NetVLAD/DataFormat/custom_format.py FREQ "path_to_original_data/query1" "path_to_data_folder" "name_folder_for_keeping_queries"

example : python GR/NetVLAD/DataFormat/custom_format.py 4 ../original_data/query/ ../data/ Query_resized

```


- For Data Loading, we use `.mat` files which contain information regarding Reference Image Paths, Query Image Paths, Ground-truth Co-ordinates for Reference and Query Images, and the Positive Localization Distance Threshold. 

```python
python GR/NetVLAD/DataFormat/custom_mat.py "path_to_data_folder" "matfile_name"


example: python GR/NetVLAD/DataFormat/custom_mat.py ../data/ SMALL

```

- To read mat file use `read_mat.py` file in `GR/NetVLAD/DataFormat/` folder passing filepath as an argument.

#### Basic run

- Extract NetVLAD Descriptors, Predictions and Cluster Masks: (you can find checkpoint file in NetVLAD/Checkpoints folder)

```python

python NetVLAD/main.py --resume 'folder_for_NetVLAD_checkpoint' --root_dir 'path_to_data_folder' --save --save_path 'save_path_for_netvlad_descriptors_and_other_things'

example: python GR/NetVLAD/NetVLAD/main.py --resume '../data/NetVLAD' \
--dataset 'SMALL' \
--root_dir '../data/' --save --save_path '../data/NetVLAD'   --nocuda
```

- One can use following command generate path lists and corresspondences for each query.
```python
python GR/NetVLAD/generate_path_lists.py --root_dir '../data/' --dataset 'SMALL' --netvlad_predictions '../data/NetVLAD/netvlad_preds.npy' --netvlad_distances '../data/NetVLAD/netvlad_dist.npy' --save_path '../data/NetVLAD'
```

- To get images pairs for query-retrieved from NetVLAD one can use below command.
```python
python GR/NetVLAD/generate_imgpairs.py ../data/NetVLAD/SMALL_netvlad_candidate_list.txt  ../data/NetVLAD/img_pairs/ ../data/Query_resized/ ../data/Reference_resized/
```




### DIR(GEM)

We have utilised official deep image retrieval repository [link](https://github.com/naver/deep-image-retrieval). Download one of the model from official DIR repo like `Resnet-101-AP-GeM.pt`.One can extract global escriptor using following command:

```python

./exec.sh input_folder_containing_all_images output_folder_for_saving_feature

example: ./exec.sh '/home/ayushsharma/Documents/College/RRC/IROS2023/seqVLAD_TRAINING_DATA/data_format/test_data/color_files_for_dir/' \
    '/home/ayushsharma/Documents/College/RRC/IROS2023/seqVLAD_TRAINING_DATA/data_format/test_data/dir_gds/'
```

Finally to generate predictions:

```python
python generate_predictions.py folder_path_for_database_images folder_path_for_query_images
example: python generate_predictions.py 'output/DIR_dbFeat.npy' 'output/DIR_qFeat.npy'

```


## HUTS loop pair detection scripts

The folder containe notebooks for performing PCA and clustering on extracted global descriptor from NetVLAD or DIR and utilising seqVLAD extract descriptor of the sequence form due to clustering for hierarchical loop detection. One can access the data used in notebook over here: [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ayush_sharma_students_iiit_ac_in/Eio9us4OsFlMoqbjrP32aMABA48zXhF1fILzwCKxlfYXNg?e=5SRqDB)

## Local Feature Matching

### RoRD


#### Basic Run


```bash
cd LFM/RoRD
```

- Selects four points in the image in the Region of interest, whose top view is required and save homography matrix. 
```python
python getRealOneGazebo.py --rgb path_to_rgb_file --depth path_to_depth_file --camera_file configs/camera_gazebo_drone.txt
 
example: python getRealOneGazebo.py --rgb ../../../original_data/color/1.jpg --depth ../../../original_data/depth/1.png --camera config/camera_habitat_original.txt 
```

- Inferring on gazebo dataset in orthographic view
    - Use following commad to generate as bash file, full of commands for rord inference on pairs given by NetVLAD like global descriptors.
        1. ` python gen_loop_pair.py ../../../data/NetVLAD/SMALL_netvlad_candidate_list.txt  ../../../data/RoRD/loop_pair.txt  ./reg_cmd.sh 5`

        2. `./reg_cmd.sh`

    - Command in the file will look like following format: (you can find RoRD model checkpoint in RoRD folder i.e. `RoRD.pth` file)
    `python register.py --rgb1 <img1.jpg>  --rgb2 <img2.jpg>  --depth1 <depth1.npy>  --depth2 <depth2.npy>  --camera_file ../configs/camera_gazebo.txt  --H ../configs/topH.npy  --model_rord ../models/rord.pth --viz3d --save_trans`

    - If homography is different for the two images, then use `--H` and `--H2` flags. You'll have in that case edit `gen_loop_pair.py` file:  
            `python register.py --H <first_homography.npy> --H2 <second_homography.npy>`  


- Following command makes a folder and saves the matches for RoRD keypoints

    `python view_best_match.py ../../../data/RoRD/rord_matches_count.txt ../../../data/RoRD/perspective/ ../../../data/RoRD/best_retrieved_for_query/  ../../../data/RoRD/query_wise`



## Backend Optimisation

### G2O


#### SE2


1. `cordTrans.py` file converts a transformation from left handed system to right handed system. But one can ignore them for habitat setting we'are using.

2. `python genG2o_fromcsv.py ../../data/small_2/Reference_Poses.csv ../../data/small_2/Query_Poses.csv ../../data/small_2/RoRD/rord_matches_count.txt  ../../data/small_2/RoRD/transition/` 

3. `python optimizePose.py ../../data/subsequences/combined/PGO/noise.g2o ../../data/subsequences/combined/PGO/loop_pairs.txt ../../data/subsequences/combined/PGO/gt.g2o`


### Backend optimisation evaluation

1. Convert `g2o` files to kitti format using `g2o_to_kitt.py` file. Right now this file only handles SE2 case. But one can make changes for SE3 case.
    `python g2o_to_kitt.py gt.g2o gt.kitti`

2. Install `evo` pypi module. Run following command to get APE/ATE:- 
    1. `evo_ape kitti gt.txt noise.txt -va --plot --plot_mode xy`
    2. `evo_ape kitti gt.txt opt.txt -va --plot --plot_mode xy`
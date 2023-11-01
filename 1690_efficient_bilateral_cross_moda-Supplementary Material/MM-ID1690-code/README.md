# Many-to-many Bilateral Cross-modality Cluster Matching (MBCCM)
Pytorch Code of MBCCM method for Cross-Modality Person Re-Identification (Visible Thermal Re-ID) on RegDB dataset [1] and SYSU-MM01 dataset [2]. 

We adopt the two-stream network structure introduced in [3]. ResNet50 is adopted as the backbone.

|Datasets    |Pretrained|  Rank@1  |   mAP    |   mINP  |  
| --------   | ------   | -------  |  ------  |  -----  |
|#RegDB      | ImageNet | ~ 83.79% | ~ 77.87% | ~65.04% |
|#SYSU-MM01  | ImageNet | ~ 53.14% | ~ 48.16% | ~32.41% |

*Both of these two datasets may have some fluctuation due to random spliting. The results might be better by finetuning the hyper-parameters. 

### 1. Prepare the datasets.

- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

    - A private download link can be requested via sending me an email (mangye16@gmail.com). 
  
- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Requirements
+ easydict==1.10
+ faiss_gpu==1.6.4
+ infomap==2.7.1
+ numpy==1.21.1
+ Pillow==9.3.0
+ Pillow==9.5.0
+ PyYAML==6.0
+ PyYAML==6.0
+ scikit_learn==1.2.0
+ scipy==1.10.1
+ tensorboardX==2.5.1
+ tensorboardX==2.6
+ torch==1.13.0
+ torchvision==0.14.0
+ tqdm==4.61.2

### 3. Training.
  Train a model on SYSU-MM01:
  ```bash
python train_mbccm.py --dataset sysu --batch-size 12 --num_pos 12 --eps 0.6 --train-iter 300 --alpha 0.9 --beta 0.5
```
Train a model on RegDB:
  ```bash
python train_mbccm.py --dataset regdb --batch-size 12 --num_pos 12 --eps 0.3 --train-iter 100 --alpha 0.9 --beta 0.5
```

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the script.

### 4. Testing.

Test a model on SYSU-MM01 or RegDB dataset by 
  ```bash
python tester.py --resume 'model_path'  --dataset sysu
```
  - `--dataset`: which dataset "sysu" or "regdb".
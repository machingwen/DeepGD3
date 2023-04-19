# Phison-defect-detection-YWL

## Requirement
```
# 將Python升級到3.8: https://tech.serhatteker.com/post/2019-12/upgrade-python38-on-ubuntu/
pip install -U scikit-learn
pip install tensorboard tensorboardX selenium twder beautifulsoup4 seaborn thop tqdm pytorch_metric_learning openpyxl natsort tensorboard_logger opencv-python pandas seaborn numpy
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install bayesian-optimization==1.4.2
pip install pycave==3.1.3
pip install ipykernel --upgrade
python3 -m ipykernel install --user
```

## Training 

- **Stage 1 : Hybrid Expert model training:**
```
python main.py --batch_size 256 --learning_rate 0.05 --seed $seed --epochs 30
```
- **Stage 2 : GMM model training**
```
python GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 5
```
- **Plotting t-SNE for Hybrid Expert Model and GMM Model**
```
python ExtractedFeatures_plotter.py --random_seed $seed --relabel --embedding_layer shared_embedding --checkpoint_path $ckpt

python GMM_Features_Ploatter.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
```
- **Bayesian optimization**
```
python BayesOpt_GMM.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
```
- **Inference**
```
CUDA_VISIBLE_DEVICES=$gpu_num python3 test_GMM_BayesOpt_GMM.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
```

## Hybrid Expert model training script
```shell
gpu_num=0
weight_path=./save/SupCon/phison_models/SupCon_phison_mobilenetv3_large_lr_0.05_decay_0.0001_bsz_256_temp_0.1_trial_0/$seed/ckpt_best.pth
for seed in 1 1212 42
do
    CUDA_VISIBLE_DEVICES=$gpu_num python main.py --batch_size 256 --learning_rate 0.05 --seed $seed --epochs 30
    CUDA_VISIBLE_DEVICES=$gpu_num python ExtractedFeatures_plotter.py --random_seed $seed --relabel --embedding_layer shared_embedding --checkpoint_path $ckpt
done
```
## GMM training script
```shell
gpu_num=0
for seed in 1 1212 42
do
    ckpt=./save/SupCon/phison_models/SupCon_phison_mobilenetv3_large_lr_0.05_decay_0.0001_bsz_256_temp_0.1_trial_0/$seed/ckpt_best.pth

    # training GMM  # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22
    for NAME in 4 8 15 18 19 20 
        do
            CUDA_VISIBLE_DEVICES=$gpu_num python GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 5
        done
    for NAME in 0 9 10 12 13 14 
        do
            CUDA_VISIBLE_DEVICES=$gpu_num python GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 50
        done
    for NAME in 1 2 3 5 6 7 11 16 17 21 22 
        do
            CUDA_VISIBLE_DEVICES=$gpu_num python GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 200
        done

    # testing Exp1 + GMM
    CUDA_VISIBLE_DEVICES=$gpu_num python3 GMM_Features_Ploatter.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
    CUDA_VISIBLE_DEVICES=$gpu_num python3 BayesOpt_GMM.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
done
```
## Inference script
```shell
gpu_num=0
for seed in 1 1212 42
do
    ckpt=./save/SupCon/phison_models/SupCon_phison_mobilenetv3_large_lr_0.05_decay_0.0001_bsz_256_temp_0.1_trial_0/$seed/ckpt_best.pth

    CUDA_VISIBLE_DEVICES=$gpu_num python3 test_GMM_BayesOpt_GMM.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
done
```


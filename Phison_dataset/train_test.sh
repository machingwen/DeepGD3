gpu_num=0
for seed in 1 1212 42
do
    ckpt=./save/SupCon/phison_models/SupCon_phison_mobilenetv3_large_lr_0.05_decay_0.0001_bsz_128_temp_0.1_trial_0/$seed/ckpt_best.pth
    CUDA_VISIBLE_DEVICES=$gpu_num python3 main.py --batch_size 128 --learning_rate 0.05 --seed $seed --epochs 30
    CUDA_VISIBLE_DEVICES=$gpu_num python3 ExtractedFeatures_Plotter.py --random_seed $seed --relabel --embedding_layer shared_embedding --checkpoint_path $ckpt
      
    for NAME in 4 8 15 18 19 20
        do
        CUDA_VISIBLE_DEVICES=$gpu_num python3 GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 5
        done

    for NAME in 0 9 10 12 13 14
        do
        CUDA_VISIBLE_DEVICES=$gpu_num python3 GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 50
        done

    for NAME in 1 2 3 5 6 7 11 16 17 21 22
        do
        CUDA_VISIBLE_DEVICES=$gpu_num python3 GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 200
        done
    CUDA_VISIBLE_DEVICES=$gpu_num python3 GMM_Features_Plotter.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
    CUDA_VISIBLE_DEVICES=$gpu_num python3 BayesOpt_GMM.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
    CUDA_VISIBLE_DEVICES=$gpu_num python3 Inference.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
done

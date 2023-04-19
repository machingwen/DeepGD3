gpu_num=0
for seed in 1 1212 42  
do
    ckpt=./save/SupCon/fruit_8_models/SupCon_fruit_8_mobilenetv3_large_lr_0.05_decay_0.0001_bsz_128_temp_0.1_trial_0/$seed/ckpt_best.pth
    CUDA_VISIBLE_DEVICES=$gpu_num python3 main.py --batch_size 128 --learning_rate 0.05 --seed $seed --epochs 30

    CUDA_VISIBLE_DEVICES=$gpu_num python3 ExtractedFeatures_plotter.py --random_seed $seed --relabel --embedding_layer shared_embedding --checkpoint_path $ckpt

    for NAME in 0 1 2 3 4 5 6 7 
        do
        CUDA_VISIBLE_DEVICES=$gpu_num python3 GMM_train.py --seed $seed --ckpt $ckpt --embedding_layer shared_embedding --componentName $NAME --gaussian_num 30
        done

    CUDA_VISIBLE_DEVICES=$gpu_num python3 GMM_Features_Ploatter.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding  
    CUDA_VISIBLE_DEVICES=$gpu_num python3 BayesOpt_GMM.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding --gaussian_num 30
    CUDA_VISIBLE_DEVICES=$gpu_num python3 Inference.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding --gaussian_num 30
done

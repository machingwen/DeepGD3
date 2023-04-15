for seed in 1212
do
    ckpt=./save/SupCon/phison_models/SupCon_phison_mobilenetv3_large_lr_0.05_decay_0.0001_bsz_128_temp_0.1_trial_0/$seed/ckpt_best.pth
    python3 show_GMM_P1_P2.py --batch_size 1024 --seed $seed --relabel --ckpt $ckpt --embedding_layer shared_embedding
done

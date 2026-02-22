# TIDS="0 1 2 3 4"
# SEEDS="0 1 2 3 4"

TIDS="0"
SEEDS="0"

for tid in $TIDS
do
for seed in $SEEDS
do
   DEFAULT="--save-model --film-aux-net mlp --last-eval-only --resolution 1mm --pretrained --epochs 1 --optimizer Adam --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th --weight-decay 0.001 --normalize"
   
   TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 age_dummy gender educat faq"

   CUDA_VISIBLE_DEVICES=0 python3 main.py --method film_demo --film-only-last --date deal --film-aux-net-act silu $TABULAR_INPUT1 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT &
   CUDA_VISIBLE_DEVICES=1 python3 main.py --method film_demo --film-only-last --date deal --film-aux-net-act silu $TABULAR_INPUT1 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=2 python3 main.py --method film_demo --film-only-last --date deal --film-aux-net-act silu $TABULAR_INPUT1 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=3 python3 main.py --method film_demo --film-only-last --date deal --film-aux-net-act silu $TABULAR_INPUT1 --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & wait


done
done


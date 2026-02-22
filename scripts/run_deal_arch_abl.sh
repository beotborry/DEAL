TIDS="0 1 2 3 4"
SEEDS="0 1 2 3 4"

for tid in $TIDS
do
for seed in $SEEDS
do
DEFAULT="--save-model --film-aux-net mlp --last-eval-only  --resolution 1mm --pretrained --epochs 30 --optimizer Adam --batch-size 4 --term 10 --data mri --mode train  --scheduler cosine_no_repeat --target ad_transition --no-best-th --weight-decay 0.001 --normalize"
TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 age_dummy gender educat faq"


CUDA_VISIBLE_DEVICES=0 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date eff_deal --mri-arch efficientnet --drop-rate 0.2 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250318_eff_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date eff_deal --mri-arch efficientnet --drop-rate 0.2 --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250318_eff_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date eff_deal --mri-arch efficientnet --drop-rate 0.2 --seed $seed --test-set-id $tid --lr 5e-5 --mri-modelpath ./trained_models/250318_eff_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date eff_deal --mri-arch efficientnet --drop-rate 0.2 --seed $seed --test-set-id $tid --lr 7e-5 --mri-modelpath ./trained_models/250318_eff_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done

for tid in $TIDS
do
for seed in $SEEDS
do
DEFAULT="--save-model --film-aux-net mlp --last-eval-only  --resolution 1mm --pretrained --epochs 30 --optimizer Adam --batch-size 4 --term 10 --data mri --mode train  --scheduler cosine_no_repeat --target ad_transition --no-best-th --weight-decay 0.001 --normalize"
TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 age_dummy gender educat faq"



CUDA_VISIBLE_DEVICES=0 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date vit_b_deal --mri-arch vit_b --seed $seed --test-set-id $tid --lr 3e-7  --mri-modelpath ./trained_models/250318_vit_b_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr1e-05_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date vit_b_deal --mri-arch vit_b --seed $seed --test-set-id $tid --lr 5e-7  --mri-modelpath ./trained_models/250318_vit_b_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr1e-05_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date vit_b_deal --mri-arch vit_b --seed $seed --test-set-id $tid --lr 1e-6  --mri-modelpath ./trained_models/250318_vit_b_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr1e-05_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date vit_b_deal --mri-arch vit_b --seed $seed --test-set-id $tid --lr 3e-6  --mri-modelpath ./trained_models/250318_vit_b_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr1e-05_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done

for tid in $TIDS
do
for seed in $SEEDS
do
DEFAULT="--save-model --film-aux-net mlp --last-eval-only  --resolution 1mm --pretrained --epochs 30 --optimizer Adam --batch-size 4 --term 10 --data mri --mode train  --scheduler cosine_no_repeat --target ad_transition --no-best-th --weight-decay 0.001 --normalize"
TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 age_dummy gender educat faq"

CUDA_VISIBLE_DEVICES=0 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date 2d_sagittal_deal --sagittal --mri-arch monai_resnet18 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250318_2d_sagittal_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date 2d_sagittal_deal --sagittal --mri-arch monai_resnet18 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250318_2d_sagittal_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date 2d_sagittal_deal --sagittal --mri-arch monai_resnet18 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250318_2d_sagittal_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py --method film_demo $TABULAR_INPUT1 --film-only-last --decouple --film-aux-net-act silu --date 2d_sagittal_deal --sagittal --mri-arch monai_resnet18 --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250318_2d_sagittal_pretrain_merge_train_val/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done
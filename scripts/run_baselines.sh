SEEDS="0 1 2 3 4"
TIDS="0 1 2 3 4"



# ERM

for tid in $TIDS
do
for seed in $SEEDS
do
   DEFAULT="--save-model --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"

   CUDA_VISIBLE_DEVICES=0 python3 main.py --date erm --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT &
   CUDA_VISIBLE_DEVICES=1 python3 main.py --date erm --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=2 python3 main.py --date erm --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=3 python3 main.py --date erm --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & wait
done
done


# ERM-T

for tid in $TIDS
do
for seed in $SEEDS
do

DEFAULT="--save-model --last-eval-only   --method tabular_only --tabular-input mmse cdrsb adas11 age gender educat faq --normalize --get-only-tabular --no-dropout-mlp --resolution 1mm --pretrained  --epochs 100 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001 "


CUDA_VISIBLE_DEVICES=0 python main.py --date tab_erm --seed $seed --test-set-id $tid --num-layer 1 --lr 0.001 $DEFAULT &
CUDA_VISIBLE_DEVICES=1 python main.py --date tab_erm --seed $seed --test-set-id $tid --num-layer 1 --lr 5e-4 $DEFAULT &
CUDA_VISIBLE_DEVICES=2 python main.py --date tab_erm --seed $seed --test-set-id $tid --num-layer 1 --lr 3e-4 $DEFAULT & 
CUDA_VISIBLE_DEVICES=3 python main.py --date tab_erm --seed $seed --test-set-id $tid --num-layer 1 --lr 1e-4 $DEFAULT & wait

done
done


# Concat-1FC, Concat-2FC

for seed in $SEEDS
do
for tid in $TIDS
do
DEFAULT="--save-model --last-eval-only   --method tab_concat --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"
TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 age gender educat faq"

CUDA_VISIBLE_DEVICES=0 python main.py $TABULAR_INPUT1 --date concat_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py $TABULAR_INPUT1 --date concat_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py $TABULAR_INPUT1 --date concat_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 1e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py $TABULAR_INPUT1 --date concat_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait

CUDA_VISIBLE_DEVICES=0 python main.py $TABULAR_INPUT1 --date concat_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py $TABULAR_INPUT1 --date concat_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py $TABULAR_INPUT1 --date concat_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 1e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py $TABULAR_INPUT1 --date concat_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done


# FiLM

for seed in $SEEDS
do
for tid in $TIDS
do
DEFAULT="--save-model  --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize --tabular-input mmse cdrsb adas11 age gender educat faq"

CUDA_VISIBLE_DEVICES=0 python main.py --date film_baseline --method film_demo --seed $seed --test-set-id $tid --lr 1e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py --date film_baseline --method film_demo --seed $seed --test-set-id $tid --lr 5e-7 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& 
CUDA_VISIBLE_DEVICES=2 python main.py --date film_baseline --method film_demo --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py --date film_baseline --method film_demo --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done


# DAFT

for seed in $SEEDS
do
for tid in $TIDS
do
DEFAULT="  --method daft --resolution 1mm --pretrained  --epochs 50 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"

CUDA_VISIBLE_DEVICES=0 python main.py --date daft_original --daft-input original --seed $seed --test-set-id $tid --lr 5e-4 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py --date daft_original --daft-input original --seed $seed --test-set-id $tid --lr 1e-3 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py --date daft_original --daft-input original --seed $seed --test-set-id $tid --lr 1e-5 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py --date daft_original --daft-input original --seed $seed --test-set-id $tid --lr 5e-6 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait

done
done

# DAFT-Ours

for seed in $SEEDS
do
for tid in $TIDS
do
DEFAULT="--save-model  --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize --tabular-input mmse cdrsb adas11 age gender educat faq"

CUDA_VISIBLE_DEVICES=0 python main.py --date daft_ours --method daft --daft-input ours --seed $seed --test-set-id $tid --lr 3e-6 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py --date daft_ours --method daft --daft-input ours --seed $seed --test-set-id $tid --lr 5e-6 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py --date daft_ours --method daft --daft-input ours --seed $seed --test-set-id $tid --lr 1e-6 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py --date daft_ours --method daft --daft-input ours --seed $seed --test-set-id $tid --lr 5e-7 --modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done

# Decouple

for tid in $TIDS
do
for seed in $SEEDS
do
   DEFAULT="--save-model --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"

   CUDA_VISIBLE_DEVICES=0 python3 main.py --date decouple --decouple --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT &
   CUDA_VISIBLE_DEVICES=1 python3 main.py --date decouple --decouple --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=2 python3 main.py --date decouple --decouple --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=3 python3 main.py --date decouple --decouple --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & wait
done
done

# SUBG

for tid in $TIDS
do
for seed in $SEEDS
do
   DEFAULT="--save-model --mri-subg --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"

   CUDA_VISIBLE_DEVICES=0 python3 main.py --date subg --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT &
   CUDA_VISIBLE_DEVICES=1 python3 main.py --date subg --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=2 python3 main.py --date subg --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=3 python3 main.py --date subg --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & wait
done
done

# Group-DRO

for tid in $TIDS
do
for seed in $SEEDS
do
for glr in 1e-4 1e-3 1e-2 1e-1
do
   DEFAULT="--save-model --group-dro --no-pos-weight --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"

   CUDA_VISIBLE_DEVICES=0 python3 main.py --date gdro --seed $seed --test-set-id $tid --lr 3e-6 --group-weight-lr $glr --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT &
   CUDA_VISIBLE_DEVICES=1 python3 main.py --date gdro --seed $seed --test-set-id $tid --lr 5e-6 --group-weight-lr $glr --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=2 python3 main.py --date gdro --seed $seed --test-set-id $tid --lr 1e-5 --group-weight-lr $glr --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=3 python3 main.py --date gdro --seed $seed --test-set-id $tid --lr 3e-5 --group-weight-lr $glr --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & wait
done
done
done

# Group-DRO + Concat

for tid in $TIDS
do
for seed in $SEEDS
do
for glr in 1e-4 1e-3 1e-2 1e-1
do
   DEFAULT="--save-model --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"

   TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 faq gender age educat"

   CUDA_VISIBLE_DEVICES=0 python3 main.py --date gdro_no_pos_weight_tab_concat --method tab_concat $TABULAR_INPUT1 --no-pos-weight --group-dro --group-weight-lr $glr --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT &
   CUDA_VISIBLE_DEVICES=1 python3 main.py --date gdro_no_pos_weight_tab_concat --method tab_concat $TABULAR_INPUT1 --no-pos-weight --group-dro --group-weight-lr $glr --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=2 python3 main.py --date gdro_no_pos_weight_tab_concat --method tab_concat $TABULAR_INPUT1 --no-pos-weight --group-dro --group-weight-lr $glr --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=3 python3 main.py --date gdro_no_pos_weight_tab_concat --method tab_concat $TABULAR_INPUT1 --no-pos-weight --group-dro --group-weight-lr $glr --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & wait
done
done
done

# SUBG + Concat

for tid in $TIDS
do
for seed in $SEEDS
do
   DEFAULT="--save-model --last-eval-only  --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"
   TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 faq gender age educat"
    
   CUDA_VISIBLE_DEVICES=0 python3 main.py --date subg_tab_concat --method tab_concat $TABULAR_INPUT1 --mri-subg --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT &
   CUDA_VISIBLE_DEVICES=1 python3 main.py --date subg_tab_concat --method tab_concat $TABULAR_INPUT1 --mri-subg --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=2 python3 main.py --date subg_tab_concat --method tab_concat $TABULAR_INPUT1 --mri-subg --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & 
   CUDA_VISIBLE_DEVICES=3 python3 main.py --date subg_tab_concat --method tab_concat $TABULAR_INPUT1 --mri-subg --seed $seed --test-set-id $tid --lr 3e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT & wait

done
done


# Attn-1FC, Attn-2FC

for seed in $SEEDS
do
for tid in $TIDS
do
for n_block in 4
do
DEFAULT="--save-model --last-eval-only   --method tab_attention --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"
TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 age gender educat faq"

CUDA_VISIBLE_DEVICES=0 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n1_n_block_${n_block} --n-attn-blocks ${n_block} --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n1_n_block_${n_block} --n-attn-blocks ${n_block} --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n1_n_block_${n_block} --n-attn-blocks ${n_block} --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 1e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n1_n_block_${n_block} --n-attn-blocks ${n_block} --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait

CUDA_VISIBLE_DEVICES=0 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n2_relu_n_block_${n_block} --n-attn-blocks ${n_block} --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n2_relu_n_block_${n_block} --n-attn-blocks ${n_block} --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n2_relu_n_block_${n_block} --n-attn-blocks ${n_block} --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 1e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py $TABULAR_INPUT1 --date tab_attention_clf_n2_relu_n_block_${n_block} --n-attn-blocks ${n_block} --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done
done


# Concat-lin-proj-1FC, Concat-lin-proj-2FC

for seed in $SEEDS
do
for tid in $TIDS
do
DEFAULT="--save-model --last-eval-only   --method tab_concat_lin_proj --resolution 1mm --pretrained  --epochs 30 --optimizer Adam  --batch-size 4 --term 10 --data mri --mode train --mri-arch monai_resnet18 --scheduler cosine_no_repeat --target ad_transition --no-best-th  --weight-decay 0.001  --normalize"
TABULAR_INPUT1="--tabular-input mmse cdrsb adas11 age gender educat faq"

CUDA_VISIBLE_DEVICES=0 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 1e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n1 --num-layer-clf 1 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait

CUDA_VISIBLE_DEVICES=0 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 5e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=1 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 1e-5 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=2 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 1e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT&
CUDA_VISIBLE_DEVICES=3 python main.py $TABULAR_INPUT1 --date concat_lin_proj_clf_n2_relu --film-aux-net-act relu --num-layer-clf 2 --seed $seed --test-set-id $tid --lr 3e-6 --mri-modelpath ./trained_models/250120_ad_pretrained/mri/ad_year3/seed0_sub_seed0_tid0_epochs100_bs4_lr0.0001_decay0.001_snp_versionmissing1_augspatial_only_aug_isNone_lamb1.0_last_no_th.pt $DEFAULT& wait
done
done
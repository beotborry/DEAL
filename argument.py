import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Fairness-aware MRI Training Pipeline"
    )

    # =========================================================
    # Basic Setup
    # =========================================================
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='Path to dataset directory')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='Execution mode')
    parser.add_argument('--set-mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='Model mode override')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--date', type=str, default='20200101',
                        help='Experiment date identifier')
    parser.add_argument('--test-set-id', type=int, default=0,
                        help='Test split identifier')

    # =========================================================
    # Data / Target
    # =========================================================
    parser.add_argument('--resolution', type=str, default='1mm',
                        choices=['1mm', '2mm'],
                        help='MRI registration resolution')
    parser.add_argument('--target', type=str, default='ad',
                        choices=['ad', 'ad_transition', 'ad_time_cox'],
                        help='Prediction target')
    parser.add_argument('--merge-train-val', action='store_true',
                        help='Merge training and validation sets')
    parser.add_argument('--normalize', action='store_true',
                        help='Age normalization for MRI')
    parser.add_argument('--get-only-tabular', action='store_true',
                        help='Return only tabular features')
    # =========================================================
    # Model Architecture
    # =========================================================
    parser.add_argument('--mri-arch', type=str, default='med_resnet18',
                        choices=['efficientnet', 'vit_t', 'vit_s', 'vit_b', 'monai_resnet18'],
                        help='MRI backbone architecture')
    parser.add_argument('--modelpath', type=str, default=None,
                        help='Path to full model checkpoint')
    parser.add_argument('--mri-modelpath', type=str, default=None,
                        help='Path to MRI backbone checkpoint')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--for-pretraining', action='store_true',
                        help='Train model for pretraining stage')

    # ---- ResNet Backbone ----
    parser.add_argument('--resnet-backbone', type=str, default='resnet_18',
                        choices=['resnet_10', 'resnet_18', 'resnet_29',
                                 'resnet_34', 'resnet_50'],
                        help='ResNet backbone type')

    # ---- EfficientNet Backbone ----
    parser.add_argument('--efficientnet-backbone', type=str, default='efficientnet-b0',
                        choices=[f'efficientnet-b{i}' for i in range(8)],
                        help='EfficientNet backbone type')

    # ---- Vision Transformer ----
    parser.add_argument('--emb-dropout', type=float, default=0.1,
                        help='Token embedding dropout rate')
    parser.add_argument('--tf-dropout', type=float, default=0.1,
                        help='Transformer dropout rate')
    parser.add_argument('--freeze-embedding', action='store_true',
                        help='Freeze embedding layer')
    parser.add_argument('--share-ln', action='store_true',
                        help='Share layer normalization')

    # =========================================================
    # Optimization
    # =========================================================
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Mini-batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of dataloader workers')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam'],
                        help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')


    parser.add_argument('--scheduler', type=str, default='cosine_no_repeat',
                        choices=['cosine_no_repeat'],
                        help='LR scheduler type')

    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--pos-weight', type=float, default=1.0,
                        help='Positive class weight')
    parser.add_argument('--no-pos-weight', action='store_true',
                        help='Disable positive class weighting')
    # =========================================================
    # MLP / Classifier
    # =========================================================
    parser.add_argument('--num-layer', type=int, default=1,
                        help='Number of MLP layers')
    parser.add_argument('--hidden-nodes', type=int, default=512,
                        help='Hidden dimension for MLP')
    parser.add_argument('--num-layer-clf', type=int, default=1,
                        help='Number of classifier layers')
    parser.add_argument('--hidden-nodes-clf', type=int, default=512,
                        help='Hidden dimension for classifier')
    parser.add_argument('--no-dropout-mlp', action='store_true',
                        help='Disable dropout in MLP')

    # =========================================================
    # Attention / Dropout
    # =========================================================
    parser.add_argument('--n-attn-blocks', type=int, default=4,
                        help='Number of attention blocks')
    parser.add_argument('--drop-rate', type=float, default=None,
                        help='Global dropout rate')

    # =========================================================
    # Fairness / Group Methods
    # =========================================================
    parser.add_argument('--mri-subg', action='store_true',
                        help='Enable SUBG')
    parser.add_argument('--group-dro', action='store_true',
                        help='Enable Group DRO')
    parser.add_argument('--group-weight-lr', type=float, default=0.01,
                        help='Group weight learning rate')
    parser.add_argument('--decouple', action='store_true',
                        help='Age-based classifier decoupling')

    # =========================================================
    # FiLM / Tabular Integration
    # =========================================================
    parser.add_argument('--film-aux-net', type=str, default='linear',
                        choices=['linear', 'mlp'],
                        help='FiLM auxiliary network type')
    parser.add_argument('--film-aux-net-act', type=str, default='silu',
                        choices=['relu', 'gelu', 'silu'],
                        help='FiLM auxiliary activation')
    parser.add_argument('--film-only-last', action='store_true',
                        help='Apply FiLM only at last layer')
    parser.add_argument('--tabular-input', action='append', nargs='+',
                        type=str, help='Tabular input feature names')
    # =========================================================
    # Training Variants
    # =========================================================
    parser.add_argument('--method', type=str, default='scratch',
                        choices=['scratch', 'feature_concat', 'film',
                                 'film_demo', 'daft',
                                 'tab_concat', 'tab_concat_lin_proj',
                                 'tab_attention', 'tabular_only'],
                        help='Training method')
    parser.add_argument('--daft-input', type=str, default='original',
                        choices=['original', 'ours'],
                        help='DAFT input type')
    parser.add_argument('--sagittal', action='store_true',
                        help='Use 2D sagittal view for MRI')

    # =========================================================
    # Logging / Evaluation
    # =========================================================
    parser.add_argument('--term', type=int, default=10,
                        help='Logging interval (epochs)')
    parser.add_argument('--last-eval-only', action='store_true',
                        help='Evaluate only last checkpoint')
    parser.add_argument('--no-best-th', action='store_true',
                        help='Do not use validation-based threshold')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model')

    args = parser.parse_args()
    
    if args.tabular_input is not None:
        args.tabular_input = [item for sublist in args.tabular_input for item in sublist]

    args.cuda = torch.cuda.is_available()
    args.resizing = True if 'vit' in args.mri_arch else False
    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')

    return args

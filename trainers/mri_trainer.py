from __future__ import print_function
from datasets.mri_dataset import MRIDataset
from trainers.generic_trainer import GenericTrainer
from os.path import join, expanduser
from networks.monai_resnet import resnet18


class Trainer(GenericTrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)

    def _get_dataset(self, args):
        mri_image_root = join(expanduser("~"), "Data/BigBrain/ADNI/masterADNI_nifti_240603_1mm/")
        train_dataset = MRIDataset(mri_image_root, split="train", seed=args.seed, resizing=args.resizing, target=args.target, year=3,
                                    for_pretraining=args.for_pretraining,
                                    subg=args.mri_subg, normalize=args.normalize, test_set_id=args.test_set_id, merge_train_val=args.merge_train_val,
                                    sagittal=args.sagittal, get_only_tabular=args.get_only_tabular)

        val_dataset = MRIDataset(mri_image_root, split="val", seed=args.seed,
                                    resizing=args.resizing,  target=args.target, year=3,
                                    subg=False, normalize=args.normalize, test_set_id=args.test_set_id, merge_train_val=args.merge_train_val,
                                    sagittal=args.sagittal, get_only_tabular=args.get_only_tabular)

        test_dataset = MRIDataset(mri_image_root, split="test", seed=args.seed, resizing=args.resizing, target=args.target, year=3, 
                                subg=args.mri_subg, normalize=args.normalize, test_set_id=args.test_set_id, merge_train_val=args.merge_train_val,
                                sagittal=args.sagittal, get_only_tabular=args.get_only_tabular)

        return train_dataset, val_dataset, test_dataset

    def _init_model(self, args):
        num_outputs = 1 if self.num_classes == 2 else self.num_classes
        if args.mri_arch == "efficientnet":
            from networks.efficientnet import EfficientNetBNwClf
            model = EfficientNetBNwClf(backbone_type=args.efficientnet_backbone, num_classes=self.num_classes,
                                        pretrained=args.pretrained, mri_modelpath=args.mri_modelpath,
                                        dropout_rate=args.drop_rate)

        elif args.mri_arch == "vit_b":
            from networks.vit_3d import ViT
            model = ViT(image_size=192, image_patch_size=16, frames=192, frame_patch_size=16,
                                num_classes=num_outputs, dim=768, depth=12, heads=12, mlp_dim=768 * 4,
                                pool='cls', channels=1, dim_head=64,
                                dropout=args.tf_dropout, emb_dropout=args.emb_dropout,
                                freeze_embedding=args.freeze_embedding, modelpath=args.mri_modelpath)

        elif args.mri_arch == "monai_resnet18":
            if args.sagittal:
                n_input_channels=3
                spatial_dims = 2
            else:
                n_input_channels=1
                spatial_dims = 3
            model = resnet18(n_input_channels=n_input_channels, num_classes=num_outputs, mri_modelpath=args.mri_modelpath,
                                decouple=args.decouple,
                                target=args.target,
                                spatial_dims=spatial_dims,
                                drop_rate=args.drop_rate)
        else:
            raise NotImplementedError(f"Architecture {args.mri_arch} not implemented")

        if args.cuda:
            model = model.cuda()
        return model
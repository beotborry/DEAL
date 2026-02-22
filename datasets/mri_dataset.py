from os.path import join, expanduser
from datasets.generic_dataset import GenericDataset
import torchio as tio
import torch
from datasets.intensity_normalization import ZNormalization
import numpy as np
from torch.nn.functional import pad


class MRIDataset(GenericDataset):
    def __init__(
            self,
            root,
            split="train",
            seed=0,
            resizing=False,
            target='ad',
            year=3,
            for_pretraining=False,
            subg=False,
            normalize=False,
            test_set_id=0,
            sagittal=False,
            resolution='1mm',
            merge_train_val=False,
            get_only_tabular=False,
    ):
        self.sagittal = sagittal
        self.subg = subg
        self.nifti_file_name = 'brain_to_MNI_syn_n4.nii.gz'

        size = (182, 218, 182) if resolution == '1mm' else (91, 109, 91)
        resizing_size = (192, 192, 192) if resolution == '1mm' else (96, 96, 96)
        intensity_norm = (ZNormalization(exclude_bg=False), tio.Clamp(-0.5, 2.5))
            
        train_transforms = (
            tio.ToCanonical(), tio.CropOrPad(resizing_size if resizing else size),
            tio.RandomFlip(),
            tio.RandomAffine(p=0.5)
        ) + intensity_norm

        test_transforms = (
            tio.ToCanonical(), tio.CropOrPad(resizing_size if resizing else size),
        ) + intensity_norm
        transform = tio.Compose(train_transforms if split=="train" else test_transforms)
        super(MRIDataset, self).__init__(mri_root=root, split=split, transform=transform, seed=seed, target=target, resolution=resolution)

        if for_pretraining:
            df_train, df_val, df_test = self._split_mri_df(
                seed=seed, test_set_id=test_set_id,  target='ad_transition', year=3,
            )
            subs_for_transition = list(df_train['SubjectID'].values) + list(df_val['SubjectID'].values) + list(df_test['SubjectID'].values)
            df_train, df_val, df_test = self._split_mri_df(
                seed=seed, test_set_id=test_set_id,
                target=target, year=3,
                subs_to_exclude=subs_for_transition, merge_train_val=merge_train_val
            )

        else:
            df_train, df_val, df_test = self._split_mri_df(
                seed=seed, test_set_id=test_set_id, target=target, year=3,
                merge_train_val=merge_train_val
            )

        self.age_mean, self.age_std = None, None
        self.normalize = normalize
        if demo_concat or normalize:
            df_train["PTAGE"] = df_train["PTAGE"].astype('float32')
            df_val["PTAGE"] = df_val["PTAGE"].astype('float32')
            df_test["PTAGE"] = df_test["PTAGE"].astype('float32')

            self.age_mean = df_train["PTAGE"].mean(axis=0)
            self.age_std = df_train["PTAGE"].std(axis=0)
            df_train["PTAGE"] = (df_train["PTAGE"] - self.age_mean) / self.age_std
            df_val["PTAGE"] = (df_val["PTAGE"] - self.age_mean) / self.age_std
            df_test["PTAGE"] = (df_test["PTAGE"] - self.age_mean) / self.age_std

            keys_to_normalize = ["PTEDUCAT", "CDRSB", "ADAS11", "MMSCORE", "FAQTOTAL",  "APOE4", "ABETA42", "TAU", "PTAU",
                                "AV45", "FDG"]
            
            for key in keys_to_normalize:
                df_train[key] = df_train[key].astype('float32')
                df_val[key] = df_val[key].astype('float32')
                df_test[key] = df_test[key].astype('float32')
                
                mean = np.nanmean(df_train[key], axis=0)
                std = np.nanstd(df_train[key], axis=0)
                std = std + 1e-8 if std == 0 else std
                df_train[key] = (df_train[key] - mean) / std
                df_val[key] = (df_val[key] - mean) / std
                df_test[key] = (df_test[key] - mean) / std
                print(f'{key} - mean: {mean:.4f} / std: {std:.4f}')

        self.gender_map = {'M': 1, 'F': 0}
        self.get_only_tabular = get_only_tabular
        self.train_group_bincount = np.bincount(df_train['group'].values)

        df_train['PTGENDER'] = df_train['PTGENDER'].map({'F': 0, 'M': 1})
        df_val['PTGENDER'] = df_val['PTGENDER'].map({'F': 0, 'M': 1})
        df_test['PTGENDER'] = df_test['PTGENDER'].map({'F': 0, 'M': 1})


        if self.split == "train":
            self.df = df_train
        elif self.split == "val":
            self.df = df_val
        else:
            self.df = df_test

        self.class_counts = self.df['Dx_num'].value_counts()
        print(f"Class counts for {self.split} split: {self.class_counts}")
        print("Group counts for {} split: {}".format(self.split, self.df["group"].value_counts()))
        self.labels = self.df["Dx_num"].values.astype(int)

        self.df.reset_index(inplace=True)

        if 'ad_transition' in target or target =='ad':
            self.groups = self.df["group"].values.astype(int)
            print("Group counts for {} split: {}".format(self.split, self.df["group"].value_counts()))
            self.group_partition = {g: np.where(self.groups == g)[0] for g in range(self.num_groups)}
            self.max_group_len = max([len(self.group_partition[g]) for g in range(self.num_groups)])

            self.base_indices = []
            self.sampling_weights = []

            for key in self.group_partition.keys():
                self.base_indices.extend(self.group_partition[key])
                self.sampling_weights.extend([self.max_group_len / len(self.group_partition[key])] * len(self.group_partition[key]))

            if self.subg:
                self.min_group_len = min([len(self.group_partition[g]) for g in range(4)])
                self.subg_indicies = []
                for key in self.group_partition.keys():
                    self.subg_indicies.extend(np.random.choice(self.group_partition[key], self.min_group_len, replace=False))

                self.base_indices = self.subg_indicies
                self.group_partition = {g: np.where(self.groups[self.subg_indicies] == g)[0] for g in range(self.num_groups)}

                print("Min size: ", self.min_group_len)
                print("Total size: ", len(self.subg_indicies))
                self.class_counts = {
                    0: len(self.group_partition[0])+len(self.group_partition[1]),
                    1: len(self.group_partition[2])+len(self.group_partition[3]),
                }

    def __getitem__(self, idx):
        if self.subg:
            idx = self.base_indices[idx]

        img_name = self.df.loc[idx, "File_name"]

        if self.target == 'ad_time_cox':
            ad_time = self.df.loc[idx, "ad_time"]
        else:
            ad_time = -1

        root = self.mri_root_dir

        if not self.get_only_tabular:
            img_path = join(root, img_name, self.nifti_file_name)
            img = tio.ScalarImage(img_path)

        group = self.df.loc[idx, "group"]

        # get demo_features
        sub_id = self.df.loc[idx, "SubjectID"]
        age = self.df.loc[idx, "PTAGE"].astype('float32')
        gender = self.df.loc[idx, "PTGENDER"]
        educat = self.df.loc[idx, "PTEDUCAT"]
        cdrsb = self.df.loc[idx, "CDRSB"]
        adas11 = self.df.loc[idx, "ADAS11"]
        mmse = self.df.loc[idx, "MMSCORE"]
        faq = self.df.loc[idx, "FAQTOTAL"]
        apoe4 = self.df.loc[idx, "APOE4"]
        abeta42 = self.df.loc[idx, "ABETA42"]
        tau = self.df.loc[idx, "TAU"]
        ptau = self.df.loc[idx, "PTAU"]
        av45 = self.df.loc[idx, "AV45"]
        fdg = self.df.loc[idx, "FDG"]

        demo_features = (sub_id, age, gender, educat, cdrsb, adas11, mmse, apoe4, abeta42, tau, ptau, av45, fdg, group, faq)
        label = self.df.loc[idx, "Dx_num"]

        if self.transform and not self.get_only_tabular:
            img = self.transform(img)
        if not self.get_only_tabular:
            img = img.data

        if self.sagittal and not self.get_only_tabular:
            sag1 = img[:, 91-3, :, :]
            sag2 = img[:, 91, :, :]
            sag3 = img[:, 91+3, :, :]

            img = torch.cat([sag1, sag2, sag3], dim=0) # shape: (3, 218, 182)
            horizontal_pad = (224-182)//2
            vertical_pad = (224-218)//2
            img = pad(img, (horizontal_pad, horizontal_pad, vertical_pad, vertical_pad))

            return img, label, group, ad_time, demo_features, idx

        if self.get_only_tabular:
            return -1, label, group, ad_time, demo_features, idx

        else:
            return img, label, group, ad_time, demo_features, idx

    def __len__(self):
        if self.subg: 
            return len(self.base_indices)
    
        return len(self.df)

from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from os.path import expanduser
from sklearn.model_selection import train_test_split


home = expanduser("~")
noisy_ad_subs = [] # the list of subjectID whose diagnosis history includes AD -> MCI

class GenericDataset(data.Dataset):
    def __init__(self, mri_root=None, split="train", transform=None, seed=0, target='ad', resolution='1mm'):
        self.mri_root_dir = mri_root
        self.num_classes = 2
        if target == "ad_time_cox":
            self.num_groups = 2
        else: self.num_groups = 4
        self.seed = seed
        self.split = split
        self.transform = transform
        self.target = target

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def _split_mri_df(self, seed=0, test_set_id=0, target='ad', year=3,
                      subs_to_exclude=None, merge_train_val=False):

        df_mri = self._get_mri_df()

        if target == 'ad_transition':
            df_mri = df_mri[df_mri['Dx_new'] == 'MCI'].copy()

            df_mri['ad_scandate'] = pd.to_datetime(df_mri['ad_scandate'])
            df_mri['last_scandate_before_ad'] = pd.to_datetime(df_mri['last_scandate_before_ad'])
            df_mri['scandate'] = pd.to_datetime(df_mri['scandate'])
            df_mri['ad_examdate'] = pd.to_datetime(df_mri['ad_examdate'])
            df_mri['last_examdate_before_ad'] = pd.to_datetime(df_mri['last_examdate_before_ad'])
            df_mri.loc[:, 'ad_time_criteria'] = df_mri['ad_examdate']
            
            df_mri = df_mri[~df_mri['ad_time_criteria'].isna()]
            df_mri.reset_index(inplace=True, drop=True)
            
            diff = (df_mri['ad_time_criteria'] - df_mri['scandate']).apply(lambda x: x.days)

            df_mri.loc[:, 'ad_time'] = diff
            mask = df_mri['ad_time_criteria'].isna()
            df_mri.loc[mask, 'ad_time'] = -1

            # convert days to int

            df_mri = df_mri[
                (df_mri['ad_transition'] == 1) | ((df_mri['ad_transition'] == 0) & (diff >= year * 365))]
            df_mri.reset_index(inplace=True, drop=True)
            df_mri.loc[:, 'before_ad_criteria'] = pd.to_datetime(df_mri['last_examdate_before_ad'])
            diff_last_scan = (df_mri['before_ad_criteria'] - df_mri['scandate']).apply(lambda x: x.days)

            # remove uncertain smci subjects among ad transtion 1 & scandate nan
            diff = (df_mri['ad_time_criteria'] - df_mri['scandate']).apply(lambda x: x.days)
            mask = (
                    (df_mri['ad_transition'] == 1) &
                    ((diff_last_scan < year * 365) & (diff >= 365 * 3))
            )
            df_mri = df_mri[~mask]
            
            df_mri.loc[:, 'before_ad_time'] = (df_mri['before_ad_criteria'] - df_mri['scandate']).apply(lambda x: x.days)
            mask = df_mri['before_ad_criteria'].isna()
            df_mri.loc[mask, 'before_ad_time'] = df_mri.loc[mask, 'ad_time']
            
            if year is None or year == 10:
                df_mri.loc[:, 'Dx_num'] = df_mri['ad_transition']
            else:
                df_mri.loc[:, 'Dx_num'] = df_mri['ad_transition'] * (
                            (df_mri['ad_time_criteria'] - df_mri['scandate']).apply(lambda x: x.days) < 365 * 3)

            # noisy subs filtering
            updated_noisy_subs_list = df_mri[(df_mri['SubjectID'].isin(noisy_ad_subs)) & (df_mri['Dx_num'] == 1)][
                'SubjectID'].to_list()
            df_mri = df_mri[~df_mri['SubjectID'].isin(updated_noisy_subs_list)]
            
        elif target == 'ad_time_cox':
            df_mri = df_mri[df_mri['Dx_new'] == 'MCI'].copy()
            
            df_mri.loc[:, 'ad_examdate'] = pd.to_datetime(df_mri.loc[:, 'ad_examdate'])
            df_mri.loc[:, 'scandate'] = pd.to_datetime(df_mri.loc[:, 'scandate'])
            df_mri.loc[:, 'ad_time'] = (df_mri.loc[:, 'ad_examdate'] - df_mri.loc[:, 'scandate']).dt.days
            df_mri.loc[df_mri['ad_transition'] == 0, 'ad_time'] = -1 * df_mri.loc[df_mri['ad_transition'] == 0, 'ad_time']
            
            df_mri = df_mri[~df_mri['SubjectID'].isin(noisy_ad_subs)]            
            df_mri.loc[:, 'Dx_num'] = -1
            
        elif target == 'ad':
            label_map, target_labels = self._get_target_labels()
            df_mri = df_mri.query("Dx_new == @target_labels")
            df_mri.loc[:, "Dx_num"] = df_mri["Dx_new"].map(lambda x: label_map[x])

        df_mri.reset_index(inplace=True, drop=True)

        if 'ad_transition' in target or target == 'ad':
            df_mri['group'] = 0
            df_mri.loc[(df_mri['Dx_num'] == 0) & (df_mri['PTAGE'] >= 75), 'group'] = 1
            df_mri.loc[(df_mri['Dx_num'] == 1) & (df_mri['PTAGE'] < 75), 'group'] = 2
            df_mri.loc[(df_mri['Dx_num'] == 1) & (df_mri['PTAGE'] >= 75), 'group'] = 3
        elif target == 'ad_time_cox':
            df_mri['group'] = 0
            df_mri.loc[(df_mri['PTAGE'] >= 75), 'group'] = 1

        if subs_to_exclude is not None:
            sub_mask = df_mri['SubjectID'].isin(subs_to_exclude)
            df_mri = df_mri[~sub_mask]
            
        home = expanduser("~")

        if target == 'ad_transition':
            file_name = "250120_df_baseline_psm_mci_split_dict.pt"
            split_subs = torch.load(join(home, f"Data/BigBrain/ADNI/{file_name}"))[test_set_id][seed]
            test_subs = split_subs["test"]
            test_mask = df_mri['SubjectID'].isin(test_subs)
            df_test = df_mri[test_mask]
            df_test.reset_index(inplace=True, drop=True)
            df_train = df_mri[~test_mask]
            df_train.reset_index(inplace=True, drop=True)

            val_subs = split_subs["val"]
            val_mask = df_train['SubjectID'].isin(val_subs)
            df_val = df_train[val_mask]
            df_val.reset_index(inplace=True, drop=True)

            if not merge_train_val:
                df_train = df_train[~val_mask]
                df_train.reset_index(inplace=True, drop=True)
                
        elif target == 'ad_time_cox':
            home = expanduser("~")
            file_name = "250120_df_baseline_psm_ad_time_cox_split_dict.pt"
            split_subs = torch.load(join(home, f"Data/BigBrain/ADNI/{file_name}"))[test_set_id][seed]
            test_subs = split_subs["test"]
            test_mask = df_mri['SubjectID'].isin(test_subs)
            df_test = df_mri[test_mask]
            df_test.reset_index(inplace=True, drop=True)
            df_train = df_mri[~test_mask]
            df_train.reset_index(inplace=True, drop=True)
            val_subs = split_subs["val"]
            val_mask = df_train['SubjectID'].isin(val_subs)
            df_val = df_train[val_mask]
            df_val.reset_index(inplace=True, drop=True)
            
            if not merge_train_val:
                df_train = df_train[~val_mask]
                df_train.reset_index(inplace=True, drop=True)
                    
        else:
            file_name = "250120_df_baseline_psm_ad_split_dict.pt"
            split_subs = torch.load(join(home, f"Data/BigBrain/ADNI/{file_name}"))[test_set_id][seed]
            test_subs = split_subs["test"]
            test_mask = df_mri['SubjectID'].isin(test_subs)
            df_test = df_mri[test_mask]
            df_test.reset_index(inplace=True, drop=True)
            df_train = df_mri[~test_mask]
            df_train.reset_index(inplace=True, drop=True)
            val_subs = split_subs["val"]
            val_mask = df_train['SubjectID'].isin(val_subs)
            df_val = df_train[val_mask]
            df_val.reset_index(inplace=True, drop=True)

            if not merge_train_val:
                df_train = df_train[~val_mask]
                df_train.reset_index(inplace=True, drop=True)

        return df_train, df_val, df_test

    @staticmethod
    def _get_mri_df():
        home = expanduser("~")
        file_name = "Data/BigBrain/ADNI/240703_df_baseline.csv"
        df_mri = pd.read_csv(join(home, file_name))

        df_mri = df_mri.rename({"Dx.new": "Dx_new"}, axis="columns")
        return df_mri

    def _split_df_by_key(self, df, key="SubjectID", ratio=0.8, seed=1, balance=True, target='ad_transition'):
        """
        Split the DataFrame based on key
        """
        idx = df[key].unique()
        np.random.seed(seed)
        np.random.shuffle(idx)
        if balance:
            if 'ad_transition' == target:
                groupwise_subs = [df[df['group'] == i][key].unique() for i in range(4)]

                for i in range(self.num_groups):
                    np.random.shuffle(groupwise_subs[i])

                # if not for_test:
                train_subs = np.concatenate([groupwise_subs[i][: int(len(groupwise_subs[i]) * ratio)] for i in range(self.num_groups)])
                test_subs = np.concatenate([groupwise_subs[i][int(len(groupwise_subs[i]) * ratio):] for i in range(self.num_groups)])

                # Check if the subjectID is disjoint between train and test
                assert len(set(train_subs).intersection(set(test_subs))) == 0

                # Split df by subjectID
                sub_idx_train = pd.Series(train_subs, name=key)
                sub_idx_test = pd.Series(test_subs, name=key)
                df_train = df.merge(sub_idx_train, how="inner", on=key)
                df_test = df.merge(sub_idx_test, how="inner", on=key)
                print(f"Train: {np.bincount(df_train['group'].values)}, Test: {np.bincount(df_test['group'].values)}")
            else:
                df_train, df_test = train_test_split(df, test_size=1-ratio, stratify=df['Dx_new'], random_state=seed)

        else:
            sub_idx_train = pd.Series(idx[: int(len(idx) * ratio)], name=key)
            sub_idx_test = pd.Series(idx[int(len(idx) * ratio):], name=key)

            df_train = df.merge(sub_idx_train, how="inner", on=key)
            df_test = df.merge(sub_idx_test, how="inner", on=key)

        return df_train, df_test

    @staticmethod
    def _get_target_labels():
        label_map = {"AD": 1, "CN": 0}
        target_labels = ["AD", "CN"]

        return label_map, target_labels

    def get_group(self, label, age):
        age_threshold = 75

        if label == 0 and age < age_threshold:
            return 0
        elif label == 0 and age >= age_threshold:
            return 1
        elif label == 1 and age < age_threshold:
            return 2
        elif label == 1 and age >= age_threshold:
            return 3

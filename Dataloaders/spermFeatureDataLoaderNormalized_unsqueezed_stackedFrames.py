import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

# from .spermFeatureDataLoaderNormalized_unsqueezed import SpermFeatureDatasetNormalizedUnsqueezed




data_csv_file = "/home/vajira/DL/Medicotask_2019/csv_files/semen_analysis_data.csv"
id_csv_file = "/home/vajira/DL/Medicotask_2019/csv_files/videos_id.csv"

data_root = "/work/vajira/data/stacked_original_frames_9x256x256/fold_1"
columns_to_return = ["Head defects (%)","Non progressive sperm motility (%)", "Immotile sperm (%)"]


def get_dataframe_of_subdirectorylist(root_dir):
    d = {'video_directory_name': [], 'file_name': []}
    df = pd.DataFrame(data=d)

    for d in os.listdir(root_dir):
        full_d = os.path.join(root_dir, d)

        for f in os.listdir(full_d):
            df2 = pd.DataFrame({'video_directory_name': [d], 'file_name': [f]})
            df = df.append(df2, ignore_index=True)

    return df


class SpermFeatureDatasetNormalizedUnsqueezed(Dataset):

    def __init__(self, csv_file_data, csv_file_id, root_dir, selected_dataColums):
        self.sperm_original_data = pd.read_csv(csv_file_data, sep=";", decimal=",")

        self.normalized_data = self.sperm_original_data.iloc[:, 1:]

        self.mean = self.normalized_data.mean()
        self.std = self.normalized_data.std()

        #print(self.std)
        #print(self.mean)

        self.normalized_data = (self.normalized_data - self.mean) / self.std
        # self.normalized_data = (self.normalized_data - self.normalized_data.min())/
        #                    (self.normalized_data.max() - self.normalized_data.min())
        # self.normalized_data = (self.normalized_data)# / 100

        self.non_nomalized_data = self.normalized_data * self.std + self.mean  # * 100 # to check the accuracy of recovering data
        # print(self.non_nomalized_data)

        self.normalized_data["ID"] = self.sperm_original_data["ID"]
        self.non_nomalized_data["ID"] = self.sperm_original_data["ID"]

        self.sperm_analysed_data = self.normalized_data  # pd.read_csv(csv_file_data, sep=";", decimal=",")
        self.sperm_video_ids = pd.read_csv(csv_file_id, sep=";")
        self.root_dir = root_dir

        self.data_df = get_dataframe_of_subdirectorylist(root_dir)
        self.data_columns = selected_dataColums

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # img_tuple = self.image_data[idx] # read image data
        data_row = self.data_df.iloc[idx]


        video_name = data_row["video_directory_name"]
        file_name = data_row["file_name"]

        data_recod_id = self.sperm_video_ids.loc[
            self.sperm_video_ids["video"] == video_name]["ID"].item()

        df_record = self.sperm_analysed_data.loc[self.sperm_analysed_data["ID"] == data_recod_id]
        df_record_non_normalized = self.non_nomalized_data.loc[self.sperm_analysed_data["ID"] == data_recod_id]

        # df_record_normalized = (df_record.iloc[:,1:] - df_record.iloc[:, 1:].mean())/df_record.iloc[:, 1:].std()
        # df_record_normalized["ID"] = df_record["ID"]
        data_values = df_record[self.data_columns].values
        data_values = np.squeeze(data_values)
        # data_values = data_values.astype('double') # .reshape(-1, 1)

        # get non-normalized data
        data_values_non_normalized = df_record_non_normalized[self.data_columns].values
        data_values_non_normalized = np.squeeze(data_values_non_normalized)


        # load pt files to return
        features_dir = os.path.join(self.root_dir, video_name)
        pt_file = os.path.join(features_dir, file_name)

        feature_tensor = torch.load(pt_file, map_location=torch.device('cpu'))  # map_location=torch.device('cpu')
        # feature_tensor = feature_tensor.unsqueeze(0) # to add a channel dimension
        feature_tensor = feature_tensor.detach()  # requires_grad(False)
        feature_tensor = feature_tensor.numpy()
    
        #print(feature_tensor.shape)
        sample = {"features": feature_tensor,
                  'data_normalized': data_values,
                  'data_non_normalized': data_values_non_normalized}

        return sample

if __name__=="__main__":
    data = SpermFeatureDatasetNormalizedUnsqueezed(data_csv_file,id_csv_file,  data_root, columns_to_return)
    m = data.mean[columns_to_return].values
    s = data.std[columns_to_return].values
    print(m)
    print(s)
    print(len(data))
    # print()
    print(data[600]['features'].shape)
    print(data[600]['data_normalized'] * s + m)
    print(data[600]['data_non_normalized'])

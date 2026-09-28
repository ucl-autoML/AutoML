import numpy as np
from sklearn.datasets import fetch_openml
from pathlib import Path

def ds_list():
    dataset_id = [31, 1464, 334, 50, 333, 1504, 3, 1494, 1510, 1489,# all 2 class no NaN
                  37, 1487, 1479, 1063, 1471, 1467, 44, 1067, 1480, 1068,# all 2 class no NaN
                  1050, 1492, 1493, 1491, 1462, 1049, 1046, 335, 151, 1485,# 7 to 2 and 3 to 100 class no NaN
                  312, 1116, 6, 1038, 1486, 1457, 1461, 1120, 1220, 4534,# 2 multi, 8 2 classes, 0 NaN
                  300, 4134, 42, 1515, 16, 14, 12, 32, 28, 18,# 9 multi, 1 2 classes, 0 NaN
                  4135, 183, 22, 54, 1501, 1468, 182, 11, 458, 15,# 8 multi, 2 2 classes, last has 16 NaNs
                  40536, 469, 188, 307, 20, 29, 1475, 1459, 1466, 46,# 8 multi, 2 2 classes, 29- 67 NaN
                  23, 4538, 375, 36, 1497, 1053, 1478, 6332, 377, 38, # 7 multi, 3 2 classes, 1053 - 25Nans  6332 - 999NaNs, 38- 6064 NaN,
                  60, 40499, 23381, 23380, 1476, 24, 451, 470, 23512, 1590, # 4 multi, 6 2 classes, 7ds w NaNs
                  2, 554, 40496, 1114, 1549, 1555, 1112, 1233, 40979, 40669, #8 multi, 2 2 classes, 3ds w NaN
                  40983, 40984, 61, 4, 40982, 7, 40701, 1554, 1552, 40966, # 7 multi, 3 2 class, 3ds w Nans
                  40994, 40670, 40975, 41027, 1553, 5, 1548, 40981, 1547, 9, # 7 multi, 3 2 class, 2ds w NaNs
                  43, 53, 1137, 1128, 1138, 1166, 1158, 1134, 1165, 1130, # 10 2 classes, 0ds w NaN
                  1139, 1145, 1161, 30, 179, 59, 40, 56, 26, 181,#3 multi, 7 2 class, 2ds w NaNs
                  55, 40900, 40971, 48, 13, 52, 10, 934, 27, 782, #3 multi, 7 2 class, 4ds w Nans
                  39, 51, 184, 41, 871, 49, 34, 35, 40910, 172, #5 multi, 5 2 class, 5 ds w NaNs
                  137, 40978, 336, 885, 171, 867, 313, 163, 875, 736, #3 multi, 7 2 class, 2ds w Nans
                  ]

    # 93 ds with 2 classes / 77 multiclass datasets
    # 33 ds with Missing values
    return dataset_id

def get_dataset(ds_id):
    data = fetch_openml(data_id=ds_id)
    return data.data, data.target

def save_dataset(dataset_id):
    for i in dataset_id:
        data = fetch_openml(data_id=i)
        ds_dir = PATH = Path(f'../ds/{i}')
        print(i)
        ds_dir.mkdir(exist_ok=True)
        np.save(ds_dir / 'X', data.data)
        np.save(ds_dir / 'Y', data.target)

def load_dataset(ds_id):
    X = np.load(f'../ds/{ds_id}/X.npy', allow_pickle=True)
    Y = np.load(f'../ds/{ds_id}/Y.npy', allow_pickle=True)
    return X, Y


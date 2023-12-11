import os
import csv

dataset_folder = '../../TikTok_dataset'
csv_path = 'TikTok_info.csv'

dataset_folder = '../../UBC_dataset'

dataset_folder = '../../UBC_dataset/train'
csv_path = 'UBC_train_info.csv'

dataset_folder = '../../UBC_dataset/test'
csv_path = 'UBC_test_info.csv'

with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['folder_id', 'folder_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, folder in enumerate(os.listdir(dataset_folder)):
        folder_path = os.path.join(dataset_folder, folder)
        writer.writerow({'folder_id': i+1, 'folder_name': folder.split(".")[0]})

# def test(csv_path):
#     with open(csv_path, 'r') as csvfile:
#         dataset = list(csv.DictReader(csvfile))

#     idx = 0  
#     video_dict = dataset[idx]
#     folder_id, folder_name = video_dict['folder_id'], video_dict['folder_name']

#     print(video_dict)
#     print(len(dataset))

# test(csv_path==csv_path)
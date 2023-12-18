import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPProcessor

# adapt from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/data/dataset.py

import torch.distributed as dist
def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


class TikTok(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=768, sample_stride=4, sample_n_frames=24,
            is_image=False, clip_model_path="openai/clip-vit-base-patch32",
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"video nums: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path,local_files_only=True)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([sample_size[0],sample_size[0]]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return self.length
    
    def get_batch(self,idx):
        video_dict = self.dataset[idx]
        folder_id, folder_name = video_dict['folder_id'], video_dict['folder_name']
        
        video_dir    =    os.path.join(self.video_folder, folder_name, f"{folder_name}.mp4")
        video_pose_dir =  os.path.join(self.video_folder, folder_name, f"{folder_name}_dwpose.mp4")
        
        video_reader = VideoReader(video_dir)
        video_reader_pose = VideoReader(video_pose_dir)
        
        
        assert len(video_reader) == len(video_reader_pose), f"len(video_reader) != len(video_reader_pose) in video {idx}"
        
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
            
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        # del video_reader
        
        pixel_values_pose = torch.from_numpy(video_reader_pose.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.
        del video_reader_pose
        
        if self.is_image:
            pixel_values = pixel_values[0]
            pixel_values_pose = pixel_values_pose[0]
        
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        
        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values
        
        pixel_values_ref_img = torch.from_numpy(ref_img.asnumpy()).permute(2, 0, 1).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.
        del video_reader
        
        # pixel_values: train objective
        # pixel_values_pose: corresponding pose
        # clip_ref_image: processed reference clip image
        # pixel_values_ref_img: ReferenceNet image
        return pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img
    
    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img = self.get_batch(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length-1)
        
        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_pose = self.pixel_transforms(pixel_values_pose)
        
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)
        pixel_values_ref_img = pixel_values_ref_img.squeeze(0)
        
        # clip_ref_image = clip_ref_image.unsqueeze(1) # [bs,1,768]
        drop_image_embeds = 1 if random.random() < 0.1 else 0

        sample = dict(
            pixel_values=pixel_values, 
            pixel_values_pose=pixel_values_pose,
            clip_ref_image=clip_ref_image,
            pixel_values_ref_img=pixel_values_ref_img,
            drop_image_embeds=drop_image_embeds,
            )
        
        return sample

class UBC_Fashion(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=768, sample_stride=4, sample_n_frames=24,
            is_image=False, clip_model_path="openai/clip-vit-base-patch32",
            is_train=True,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"video nums: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        self.is_train = is_train
        self.spilt = 'train' if self.is_train else 'test'
        
        self.clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path,local_files_only=True)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([sample_size[0],sample_size[0]]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def __len__(self):
        return self.length
    
    def get_batch(self,idx):
        video_dict = self.dataset[idx]
        folder_id, folder_name = video_dict['folder_id'], video_dict['folder_name']
        
        video_dir    =    os.path.join(self.video_folder, self.spilt, f"{folder_name}.mp4")
        video_pose_dir =  os.path.join(self.video_folder, self.spilt+"_dwpose", f"{folder_name}.mp4")
        
        video_reader = VideoReader(video_dir)
        video_reader_pose = VideoReader(video_pose_dir)
        
        
        assert len(video_reader) == len(video_reader_pose), f"len(video_reader) != len(video_reader_pose) in video {idx}"
        
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]
            
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        # del video_reader
        
        pixel_values_pose = torch.from_numpy(video_reader_pose.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.
        del video_reader_pose
        
        if self.is_image:
            pixel_values = pixel_values[0]
            pixel_values_pose = pixel_values_pose[0]
        
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        
        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values
        
        pixel_values_ref_img = torch.from_numpy(ref_img.asnumpy()).permute(2, 0, 1).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.
        del video_reader
        
        # pixel_values: train objective
        # pixel_values_pose: corresponding pose
        # clip_ref_image: processed reference clip image
        # pixel_values_ref_img: ReferenceNet image
        return pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img
    
    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, pixel_values_pose, clip_ref_image, pixel_values_ref_img = self.get_batch(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length-1)
        
        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values_pose = self.pixel_transforms(pixel_values_pose)
        
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img)
        pixel_values_ref_img = pixel_values_ref_img.squeeze(0)
        
        # clip_ref_image = clip_ref_image.unsqueeze(1) # [bs,1,768]
        drop_image_embeds = 1 if random.random() < 0.1 else 0
        sample = dict(
            pixel_values=pixel_values, 
            pixel_values_pose=pixel_values_pose,
            clip_ref_image=clip_ref_image,
            pixel_values_ref_img=pixel_values_ref_img,
            drop_image_embeds=drop_image_embeds,
            )
        
        return sample




# https://github.com/tencent-ailab/IP-Adapter/blob/main/tutorial_train.py#L341

def collate_fn(data):
    # pixel_values = torch.cat([example["pixel_values"] for example in data], dim=0)
    # pixel_values_pose = torch.cat([example["pixel_values_pose"] for example in data], dim=0)
    # clip_ref_image = torch.stack([example["clip_ref_image"] for example in data])
    # pixel_values_ref_img = torch.cat([example["pixel_values_ref_img"] for example in data], dim=0)
    
    pixel_values = torch.stack([example["pixel_values"] for example in data])
    pixel_values_pose = torch.stack([example["pixel_values_pose"] for example in data])
    clip_ref_image = torch.cat([example["clip_ref_image"] for example in data])
    pixel_values_ref_img = torch.stack([example["pixel_values_ref_img"] for example in data])
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.Tensor(drop_image_embeds)
    
    return {
        "pixel_values": pixel_values,
        "pixel_values_pose": pixel_values_pose,
        "clip_ref_image": clip_ref_image,
        "pixel_values_ref_img": pixel_values_ref_img,
        "drop_image_embeds": drop_image_embeds,
    }


if __name__ == "__main__":
    # from animatediff.utils.util import save_videos_grid

    dataset = TikTok(
        csv_path="./data/TikTok_info.csv",
        video_folder="./TikTok_dataset",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
        clip_model_path = "./pretrained_models/clip-vit-base-patch32"
    )    
    
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn,batch_size=4, num_workers=0,)
    # dataloader = torch.utils.data.DataLoader(dataset,batch_size=1, num_workers=0,)

    for idx, batch in enumerate(dataloader):
        print(idx)
        # print(batch["pixel_values"])
        print(batch["pixel_values"].size())
        print(batch["pixel_values_pose"].size())
        print(batch["clip_ref_image"].size())
        print(batch["pixel_values_ref_img"].size())
        print(batch["drop_image_embeds"].size()) # torch.Size([4])
        break

    # python3 -m data.dataset
        



# if __name__ == "__main__":
#     # from animatediff.utils.util import save_videos_grid

#     dataset = WebVid10M(
#         csv_path="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/results_2M_val.csv",
#         video_folder="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/2M_val",
#         sample_size=256,
#         sample_stride=4, sample_n_frames=16,
#         is_image=True,
#     )
#     import pdb
#     pdb.set_trace()
    
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
#     for idx, batch in enumerate(dataloader):
#         print(batch["pixel_values"].shape, len(batch["text"]))
#         # for i in range(batch["pixel_values"].shape[0]):
#         #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)

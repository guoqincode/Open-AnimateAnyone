import torch

ckpt_path = './outputs/train_stage_1-2023-12-15T21-25-59/checkpoints/checkpoint-epoch-2400.ckpt'

full_state_dict = torch.load(ckpt_path,map_location='cpu')


poseguider_state_dict = full_state_dict['poseguider_state_dict']
referencenet_state_dict = full_state_dict['referencenet_state_dict']
unet_state_dict = full_state_dict['unet_state_dict']

poseguider_ckpt_path = 'pretraind_models/poseguider_stage_1.ckpt'
referencenet_ckpt_path = 'pretraind_models/referencenet_stage_1.ckpt'
unet_ckpt_path = 'pretraind_models/unet_stage_1.ckpt'


torch.save(poseguider_state_dict, poseguider_ckpt_path)
torch.save(referencenet_state_dict, referencenet_ckpt_path)
torch.save(unet_state_dict, unet_ckpt_path)

from share import *
import os 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from omegaconf import OmegaConf

#--------------------------------------------> Configs
batch_size = 10
logger_freq = 300
learning_rate = 1e-5 #1e-5
sd_locked = True 
only_mid_control = False
max_epochs=500
num_workers=16
every_n_epochs=4
default_root_dir= './1.model_save'
model_name='training_model_name' #training_model_name
resume_path='./your checkpoints'
yaml_config='./models/lightdiff_v15.yaml'

#--------------------------------------------> Configs



default_root_dir = os.path.join(default_root_dir, model_name)
if not os.path.exists(default_root_dir):
    os.makedirs(default_root_dir)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.

# condition_select = OmegaConf.load(yaml_config).condition['condition_select']
model = create_model(yaml_config).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
# dataset = MyDataset(condition_select)
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

trainer = pl.Trainer(strategy='ddp',gpus=[1,2], precision=32, callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=every_n_epochs, save_top_k=-1), logger], default_root_dir=default_root_dir, max_epochs=max_epochs)


# Train!
trainer.fit(model, dataloader)



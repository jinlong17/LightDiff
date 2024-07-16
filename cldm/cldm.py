import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config,default
from ldm.models.diffusion.ddim import DDIMSampler


#--------------------------------------------->LDRM and Multi-Condition Adapter
import os
from argparse import ArgumentParser

from cldm.LDRM import MMDLoss

import pytorch_lightning as pl

# from BEVDepth.bevdepth.callbacks.ema import EMACallback
from BEVDepth.bevdepth.utils.torch_dist import all_gather_object, synchronize
from BEVDepth.bevdepth.exps.nuscenes.base_exp import BEVDepthLightningModel
import torch 

mmdloss = MMDLoss(4)

class MC_Adapter(nn.Module):
    """ Channel attention module  from https://github.com/junfu1115/DANet/"""
    def __init__(self, in_dim):
        super(MC_Adapter, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    

class BEVDepthConfig:
    def __init__(self, extra_trainer_config_args=None, exp_name='base_exp'):
        if extra_trainer_config_args is None:
            extra_trainer_config_args = {}
        self.extra_trainer_config_args = extra_trainer_config_args
        self.exp_name = exp_name
        self.args = self.parse_args()
        
    def parse_args(self):
        parent_parser = ArgumentParser(add_help=False)
        parent_parser = pl.Trainer.add_argparse_args(parent_parser)
        parent_parser.add_argument('-e',
                                   '--evaluate',
                                   dest='evaluate',
                                   action='store_true',
                                   default=True,
                                   help='evaluate model on validation set')
        parent_parser.add_argument('-p',
                                   '--predict',
                                   dest='predict',
                                   action='store_true',
                                   help='predict model on testing set')
        parent_parser.add_argument('-b', '--batch_size_per_device', default=1, type=int)
        parent_parser.add_argument('--seed',
                                   type=int,
                                   default=1,
                                   help='seed for initializing training.')
        parent_parser.add_argument('--ckpt_path',
                                   default='/BEVDepth/BEVDepth_checkpoint/bev_depth_lss_r50_256x704_128x128_24e_2key.pth',  
                                   type=str)

        parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
        parser.set_defaults(profiler='simple',
                            deterministic=False,
                            max_epochs=self.extra_trainer_config_args.get('epochs', 24),
                            accelerator='ddp',
                            num_sanity_val_steps=0,
                            gradient_clip_val=5,
                            limit_val_batches=0,
                            enable_checkpointing=True,
                            precision=16,
                            default_root_dir=os.path.join('./outputs/', self.exp_name))
        args = parser.parse_args()
        args.gpus = 1
        return args
    
    def create_model(self):
        return BEVDepthLightningModel(**vars(self.args))


config = BEVDepthConfig()
bevdepth_model = config.create_model()

#--------------------------------------------->LDRM and Multi-Condition Adapter




class ControlledUnetModel(UNetModel):

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            bl_condition_num = 1,
            bl_condition_channel = (3),
            with_condition_attention = False,
            with_gamma = False
    ):
        super().__init__()

        ###TODO:added by baolu
        self.bl_condition_num = bl_condition_num
        self.bl_condition_channel = bl_condition_channel
        self.with_condition_attentation = with_condition_attention
        self.with_gamma = with_gamma
        if self.with_condition_attentation == True:
            self.condition_attn = MC_Adapter(in_dim = 320 * self.bl_condition_num)
            
        if self.with_gamma == True:
            self.gamma = nn.Parameter(torch.randn(320,64,64),requires_grad=True)



        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        if self.bl_condition_num == 1:
            self.input_hint_block = TimestepEmbedSequential(
                #conv_nd(dims, bl_condition_channel, 16, 3, padding=1),
                conv_nd(dims, 3, 16, 3, padding=1),###########################################################
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )
        elif self.bl_condition_num == 2:
            self.input_hint_block_1 = TimestepEmbedSequential(
                conv_nd(dims, bl_condition_channel[0], 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )
            self.input_hint_block_2 = TimestepEmbedSequential(
                conv_nd(dims, bl_condition_channel[1], 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )
        elif self.bl_condition_num == 3:
            self.input_hint_block_1 = TimestepEmbedSequential(
                conv_nd(dims, bl_condition_channel[0], 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )
            self.input_hint_block_2 = TimestepEmbedSequential(
                conv_nd(dims, bl_condition_channel[1], 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )
            self.input_hint_block_3 = TimestepEmbedSequential(
                conv_nd(dims, bl_condition_channel[2], 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)


        ###TODO:added by baolu
        if self.bl_condition_num == 1:
            guided_hint = self.input_hint_block(hint, emb, context)

        elif self.bl_condition_num == 2:
            hint_1 = hint[:,0:self.bl_condition_channel[0],:,:]
            hint_2 = hint[:,self.bl_condition_channel[0]:self.bl_condition_channel[0]+self.bl_condition_channel[1],:,:]

            guided_hint_1 = self.input_hint_block_1(hint_1, emb, context)
            guided_hint_2 = self.input_hint_block_2(hint_2, emb, context)

            guided_hint = guided_hint_1 + guided_hint_2
        
        elif self.bl_condition_num == 3:
            hint_1 = hint[:,0:self.bl_condition_channel[0],:,:]
            hint_2 = hint[:,self.bl_condition_channel[0]:self.bl_condition_channel[0]+self.bl_condition_channel[1],:,:]
            hint_3 = hint[:,self.bl_condition_channel[0]+self.bl_condition_channel[1]:self.bl_condition_channel[0]+self.bl_condition_channel[1]+self.bl_condition_channel[2],:,:]

            guided_hint_1 = self.input_hint_block_1(hint_1, emb, context)
            guided_hint_2 = self.input_hint_block_2(hint_2, emb, context)
            guided_hint_3 = self.input_hint_block_3(hint_3, emb, context)

            #320
            if self.with_condition_attentation == True:
                guided_hint_cat = torch.cat((guided_hint_1,guided_hint_2,guided_hint_3),dim=1)
                guided_hint_cat_after_attn = self.condition_attn(guided_hint_cat)
                guided_hint_1 = guided_hint_cat_after_attn[:,0:320,:,:]
                guided_hint_2 = guided_hint_cat_after_attn[:,320:640,:,:]
                guided_hint_3 = guided_hint_cat_after_attn[:,640:960,:,:]


            if self.with_gamma == True:
                bs = hint.shape[0]
                gamma = self.gamma.repeat(bs,1,1,1)
                guided_hint = guided_hint_1 + guided_hint_2 + guided_hint_3 + gamma
            else:
                guided_hint = guided_hint_1 + guided_hint_2 + guided_hint_3 


        #guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    ###TODO:jinlong:  added loss of BEVdepth

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)

        #add the detection loss
        x_noisy, model_output = loss[2], loss[3]
        img_x = x_noisy-model_output
        step = loss[4].item()


        detection_loss, depth_loss = self.bevdepth_cal(batch, c, img_x, **kwargs)

        mmd_loss = self.mmd_cal(batch, c, img_x, **kwargs)

        ###############################################
        scale_det = 0
        scale_depth = 1
        scale_mmd = 1
        ###############################################

        if step<=50:
            total_loss = loss[0]*(detection_loss*scale_det + depth_loss*scale_depth) + mmd_loss*scale_mmd
            print('step: ', step, 'bev_det_loss :   ', detection_loss.item()*scale_det,'  depth_loss: ', depth_loss.item()*scale_depth, 'mmd_loss :  ', mmd_loss.item()*scale_mmd)
        else:
            total_loss = loss[0]


        return [total_loss,loss[1]]


    def mmd_cal(self, batch, cond, img_x, **kwargs):
        # batch_size=2
        N=4

        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        mmd_loss = mmdloss(img_x, z)

        return mmd_loss


    def bevdepth_cal(self, batch, cond, img_x, **kwargs):
        batch_size=2
        N=4
        n_row=2
        ddim_steps=50
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        n_row = min(z.shape[0], n_row)

        ####0) mats, gt_boxes, gt_labels, depth_labels for detection loss
        # mats, gt_boxes, gt_labels, depth_labels = batch[-4], batch[-3], batch[-2], batch[-1]
        mats, gt_boxes, gt_labels, depth_labels = batch['mats'],batch['gt_boxes'],batch['gt_labels'],batch['depth_labels']


        ####1) generate the img: samples
        # ddim_sampler = DDIMSampler(self)
        samples = img_x
        batch_size, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        # samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        x_samples = self.decode_first_stage(samples)

        ###2) Prepare the six imgs：
        from torchvision.transforms.functional import resize as tv_resize
        from torchvision.transforms.functional import crop as tv_crop
        device_gpu = x_samples.device
        bevdepth_model.to(device_gpu)
        mmdloss.to(device_gpu)


        target_sample = x_samples
        target_sample = tv_resize(target_sample, (396, 704))
        target_sample = tv_crop(target_sample, 140, 0, 256, 704)
        target_sample = target_sample.reshape(batch_size,1,1,3, 256, 704)
        sweep_imgs=torch.zeros(batch_size, 1, 6, 3, 256, 704).to(device_gpu)
        sweep_imgs[:,:,1:2,:,:,:] = target_sample



        ####2) calculate the detection loss
        depth_labels = depth_labels.squeeze()

        new_mats = {}
        for key, value in mats.items():
            #
            modified_value = value.squeeze(1)
            new_mats[key] = modified_value
            d_num,d_w, d_h =  depth_labels.shape
            # d_w, d_h =  depth_labels.shape
        depth_labels_z=torch.zeros_like(depth_labels).to(device_gpu)
        depth_labels_z[1:2,:,:] = depth_labels[1:2,:,:] 
        depth_labels_new = depth_labels_z.reshape(batch_size,1,6,d_w, d_h)
        input = (sweep_imgs, new_mats, 1, 1, gt_boxes, gt_labels, depth_labels_new)

        with torch.no_grad():
            
            detection_loss, depth_loss = bevdepth_model.training_step(input)

        return detection_loss, depth_loss


    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        # return loss, loss_dict
        return loss, loss_dict, x_noisy, model_output, t


    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

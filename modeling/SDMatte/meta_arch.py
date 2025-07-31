import torch
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding
from utils import replace_unet_conv_in, replace_attention_mask_method, add_aux_conv_in
from utils.replace import CustomUNet
import random

AUX_INPUT_DIT = {
    "auto_mask": "auto_coords",
    "point_mask": "point_coords",
    "bbox_mask": "bbox_coords",
    "mask": "mask_coords",
    "trimap": "trimap_coords",
}


class SDMatte(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        conv_scale=3,
        num_inference_steps=1,
        aux_input="bbox_mask",
        use_aux_input=False,
        use_coor_input=True,
        use_dis_loss=True,
        use_attention_mask=True,
        use_encoder_attention_mask=False,
        add_noise=False,
        attn_mask_aux_input=["point_mask", "bbox_mask", "mask"],
        aux_input_list=["point_mask", "bbox_mask", "mask"],
        use_encoder_hidden_states=True,
        residual_connection=False,
        use_attention_mask_list=[True, True, True],
        use_encoder_hidden_states_list=[True, True, True],
    ):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.unet = CustomUNet.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True, ignore_mismatched_sizes=False
        )
        self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.num_inference_steps = num_inference_steps
        self.aux_input = aux_input
        self.use_aux_input = use_aux_input
        self.use_coor_input = use_coor_input
        self.use_dis_loss = use_dis_loss
        self.use_attention_mask = use_attention_mask
        self.use_encoder_attention_mask = use_encoder_attention_mask
        self.add_noise = add_noise
        self.attn_mask_aux_input = attn_mask_aux_input
        self.aux_input_list = aux_input_list
        self.use_encoder_hidden_states = use_encoder_hidden_states
        if use_encoder_hidden_states:
            self.unet = add_aux_conv_in(self.unet)
        if not add_noise:
            conv_scale -= 1
        if not use_aux_input:
            conv_scale -= 1
        if conv_scale > 1:
            self.unet = replace_unet_conv_in(self.unet, conv_scale)
        replace_attention_mask_method(self.unet, residual_connection)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.train()
        self.unet.use_attention_mask_list = use_attention_mask_list
        self.unet.use_encoder_hidden_states_list = use_encoder_hidden_states_list

    def forward(self, data):
        rgb = data["image"].cuda()
        B = rgb.shape[0]

        if self.aux_input is None and self.training:
            aux_input_type = random.choice(self.aux_input_list)
        elif self.aux_input is None:
            aux_input_type = "point_mask"
        else:
            aux_input_type = self.aux_input

        # get aux input latent
        if self.use_aux_input:
            aux_input = data[aux_input_type].cuda()
            aux_input = aux_input.repeat(1, 3, 1, 1)
            aux_input_h = self.vae.encoder(aux_input.to(rgb.dtype))
            aux_input_moments = self.vae.quant_conv(aux_input_h)
            aux_input_mean, _ = torch.chunk(aux_input_moments, 2, dim=1)
            aux_input_latent = aux_input_mean * self.vae.config.scaling_factor
        else:
            aux_input_latent = None

        # get aux coordinate
        coor_name = AUX_INPUT_DIT[aux_input_type]
        coor = data[coor_name].cuda()
        if coor_name == "point_coords":
            N = coor.shape[1]
            for i in range(N, 1680):
                if 1680 % i == 0:
                    num_channels = 1680 // i
                    pad_size = i - N
                    padding = torch.zeros((B, pad_size), dtype=coor.dtype, device=coor.device)
                    coor = torch.cat([coor, padding], dim=1)
                    zero_coor = torch.zeros((B, pad_size + N), dtype=coor.dtype, device=coor.device)
                    break
            if self.use_coor_input:
                coor = get_timestep_embedding(
                    coor.flatten(),
                    num_channels,
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0,
                )
            else:
                coor = get_timestep_embedding(
                    zero_coor.flatten(),
                    num_channels,
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0,
                )
            added_cond_kwargs = {"point_coords": coor}
        else:
            if self.use_coor_input:
                added_cond_kwargs = {"bbox_mask_coords": coor}
            else:
                coor = torch.tensor([[0, 0, 1, 1]] * B).cuda()
                added_cond_kwargs = {"bbox_mask_coords": coor}

        # get attention mask
        if self.use_attention_mask and aux_input_type in self.attn_mask_aux_input:
            attention_mask = data[aux_input_type].cuda()
            attention_mask = (attention_mask + 1) / 2
            attention_mask = F.interpolate(attention_mask, scale_factor=1 / 8, mode="nearest")
            attention_mask = attention_mask.flatten(start_dim=1)
        else:
            attention_mask = None

        # encode rgb to latents
        rgb_h = self.vae.encoder(rgb)
        rgb_moments = self.vae.quant_conv(rgb_h)
        rgb_mean, _ = torch.chunk(rgb_moments, 2, dim=1)
        rgb_latent = rgb_mean * self.vae.config.scaling_factor

        # get encoder_hidden_states
        if self.use_encoder_hidden_states and aux_input_latent is not None:
            encoder_hidden_states = self.unet.aux_conv_in(aux_input_latent)
            encoder_hidden_states = encoder_hidden_states.view(B, 1024, -1)
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)

        if "caption" in data:
            prompt = data["caption"]
        else:
            prompt = [""] * B
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to("cuda")
        text_embed = self.text_encoder(text_input_ids)[0]
        encoder_hidden_states_2 = text_embed

        # get class_label
        is_trans = data["is_trans"].cuda()
        trans = 1 - is_trans

        # get timesteps
        timestep = torch.tensor([1], device="cuda").long()

        # unet
        unet_input = torch.cat([rgb_latent, aux_input_latent], dim=1)
        label_latent = self.unet(
            sample=unet_input,
            trans=trans,
            timestep=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_2=encoder_hidden_states_2,
            added_cond_kwargs=added_cond_kwargs,
            attention_mask=attention_mask,
        ).sample
        label_latent = label_latent / self.vae.config.scaling_factor
        z = self.vae.post_quant_conv(label_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        label_mean = stacked.mean(dim=1, keepdim=True)
        output = torch.clip(label_mean, -1.0, 1.0)
        output = (output + 1.0) / 2.0
        return output

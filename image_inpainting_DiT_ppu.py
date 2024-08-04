import torch
import torch.nn as nn
import torchvision
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import argparse
import torch.utils.checkpoint as checkpoint
import os, shutil
from PIL import Image
import time
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel
from download import find_model
from models import DiT_models, DiT
import numpy as np
import random
import time

def clean_folder(output_path):
    # Check if the folder exists
    if os.path.exists(output_path) and os.path.isdir(output_path):
        # Iterate over all the files and folders in the specified directory
        for filename in os.listdir(output_path):
            file_path = os.path.join(output_path, filename)
            try:
                # Check if it is a file and then remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it is a directory and then remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {output_path} does not exist.')


def inpainting_loss_fn(target_image, device=None, torch_dtype=torch.float32):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transform the image to tensor and move to the specified device
    target_image = torchvision.transforms.ToTensor()(target_image).to(device, dtype=torch_dtype)

    # Define the mask generation
    # def generate_random_square_mask(image, mask_ratio=0.2):
    #     _, height, width = image.shape
    #     mask_size = int((height * width * mask_ratio) ** 0.5)
    #     x_start = random.randint(0, width - mask_size)
    #     y_start = random.randint(0, height - mask_size)
    #     mask = torch.ones_like(image).to(device, dtype=torch_dtype)
    #     mask[:, y_start:y_start + mask_size, x_start:x_start + mask_size] = 0
    #     return mask
    
    

    # # Generate a random mask to mask 20% of the image as a random square area
    # mask = generate_random_square_mask(target_image, mask_ratio=0.2)
    
    # randomly mask 90% of the image
    mask = torch.rand((1,256,256), dtype=torch.float32).to(device, dtype=torch_dtype)
    mask = mask > 0.9
    
    # Create the masked image
    masked_image = target_image * mask
    masked_image = masked_image.to(device, dtype=torch_dtype).unsqueeze(0)
    
    def loss_fn(im_pix_un, prompts=None):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        # Apply the generated mask to the predicted image
        masked_im_pix = im_pix * mask
        # Compute the inpainting loss as the mean squared error
        loss = torch.sum(torch.abs(masked_image - masked_im_pix))
        return loss
        
    return loss_fn, masked_image[0]



class SequentialDDIM:

    def __init__(self, timesteps = 100, eta = 0.0, cfg_scale = 4.0, device = "cuda", use_fp16 = True):
        self.eta = eta 
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.diffusion = create_diffusion(str(timesteps))
        self.device = device
        self.cfg_scale = cfg_scale
        self.use_fp16 = use_fp16

        # compute some coefficients in advance
        now_coeff = torch.tensor(1 - self.diffusion.alphas_cumprod, dtype = torch.float32)
        next_coeff = torch.tensor(1 - self.diffusion.alphas_cumprod_prev, dtype = torch.float32)
        now_coeff = torch.clamp(now_coeff, min = 0)
        next_coeff = torch.clamp(next_coeff, min = 0)
        m_now_coeff = torch.clamp(1 - now_coeff, min = 0)
        m_next_coeff = torch.clamp(1 - next_coeff, min = 0)
        self.noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
        self.nl = self.noise_thr * self.eta
        self.nl[0] = 0.
        m_nl_next_coeff = torch.clamp(next_coeff - self.nl**2, min = 0)
        self.coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
        self.coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * self.coeff_x

    def wrap_model(self, model):
        return self.diffusion._wrap_model(model.forward_with_cfg)

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0]

    def prepare_model_kwargs(self, class_cond = None):
        t = self.num_steps - len(self._samples)
        model_kwargs = {
            "x": torch.stack([self._samples[0], self._samples[0]]),
            "ts": torch.tensor([t, t], device = self.device),
            "y": None if class_cond is None else torch.tensor([class_cond, 1000], device = self.device),
            "cfg_scale": self.cfg_scale
        }

        if self.use_fp16:
            model_kwargs["x"] = model_kwargs["x"].to(dtype = torch.float16)
        else:
            model_kwargs["x"] = model_kwargs["x"].to(dtype = torch.float32)

        return model_kwargs


    def step(self, model_output):
        model_output, _ = torch.split(model_output, 4, dim=1)
        model_output, _ = model_output.chunk(2, dim=0)
        direction = model_output[0]
        direction = direction.to(dtype = torch.float32)

        t = self.num_steps - len(self._samples)

        assert t >= 0

        now_sample = self.coeff_x[t] * self._samples[0] + self.coeff_d[t] * direction  + self.nl[t] * self.noise_vectors[t]

        self._samples.insert(0, now_sample)
        
        if len(self._samples) > self.timesteps:
            self._is_finished = True

    def initialize(self, noise_vectors):
        self._is_finished = False

        self.noise_vectors = noise_vectors

        self._samples = [self.noise_vectors[-1]]
      

def sequential_sampling(model, sampler, class_cond, noise_vectors): 
    wrapped_model = sampler.wrap_model(model)

    sampler.initialize(noise_vectors)

    model_time = 0
    step = 0
    while not sampler.is_finished():
        step += 1
        model_kwargs = sampler.prepare_model_kwargs(class_cond = class_cond)
        model_output = checkpoint.checkpoint(wrapped_model, **model_kwargs,  use_reentrant=False)
        sampler.step(model_output) 

    return sampler.get_last_sample()


def decode_latent(decoder, latent):
    img = decoder.decode(latent.unsqueeze(0) / 0.18215).sample
    return img

def to_img(img):
    img = torch.clamp(127.5 * img.cpu().float() + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()

    return img[0]

def main():
    parser = argparse.ArgumentParser(description='Diffusion Optimization with Differentiable Objective')
    parser.add_argument('--model', type=str, default="/mnt/workspace/workgroup/tangzhiwei.tzw/sdv1-5-full-diffuser", help='path to the model')
    parser.add_argument('--num_steps', type=int, default=50, help='number of steps for optimization')
    parser.add_argument('--eta', type=float, default=1.0, help='noise scale')
    parser.add_argument('--guidance_scale', type=float, default=4.0, help='guidance scale')
    parser.add_argument('--device', type=str, default="cuda", help='device for optimization')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--opt_steps', type=int, default=10000, help='number of optimization steps')
    parser.add_argument('--opt_time', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=1, help='log interval')
    parser.add_argument('--precision', choices = ["bf16", "fp16", "fp32"], default="fp32", help='precision for optimization')
    parser.add_argument('--gamma', type=float, default=0., help='mean penalty')
    parser.add_argument('--subsample', type=int, default=1, help='subsample factor')
    parser.add_argument('--lr', type=float, default=0.1, help='stepsize for optimization')
    parser.add_argument('--target_image', type=str, help="path to the target image")
    args = parser.parse_args()

    latent_size = 256 // 8
    device = args.device
    model = DiT_models["DiT-XL/2"](
            input_size=latent_size,
            num_classes=1000
        ).to(device)
    model.load_state_dict(find_model("/mnt/workspace/workgroup/tangzhiwei.tzw/DiT-XL-2-256x256.pt"))
        
    vae = AutoencoderKL.from_pretrained("/mnt/workspace/workgroup/tangzhiwei.tzw/sdv1-5-full-diffuser/vae").to(device)

    # clear output path
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_path = f"output_ppu/{now_time}"
    os.makedirs(output_path, exist_ok=True)

    # load target image
    target_image = Image.open(args.target_image).convert("RGB")
    class_cond = int(args.target_image.split("_")[-1].strip('.png'))
    img_name = args.target_image.split("/")[-1]

    loss_fn, masked_image = inpainting_loss_fn(torch_dtype = torch.float32, device = args.device, target_image = target_image)
    
    # save masked image
    masked_img = Image.fromarray((masked_image * 255).permute(1, 2, 0).cpu().to(dtype=torch.uint8).numpy())
    masked_img.save(os.path.join(output_path, f"masked_image_{img_name}"))
    
    args.seed = args.seed + 100

    torch.manual_seed(args.seed)
    

    noise_vectors = torch.randn(args.num_steps + 1, 4, 32, 32, device = args.device)
   

    noise_vectors.requires_grad_(True)
    optimize_groups = [{"params":noise_vectors, "lr":args.lr}]

    optimizer = torch.optim.AdamW(optimize_groups)

    best_reward = -1e9
    best_sample = None

    # start optimization
    use_amp = False if args.precision == "fp32" else True
    grad_scaler = GradScaler(enabled=use_amp, init_scale = 8192)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    
    if args.eta > 0:
        dim = len(noise_vectors[:(args.opt_time + 1)].flatten())
    else:
        dim = len(noise_vectors[-1].flatten())

    subsample_dim = round(4 ** args.subsample)
    subsample_num = dim // subsample_dim

    print("="*20)
    print(subsample_dim, subsample_num)
    print("="*20)

    for i in range(args.opt_steps):
        optimizer.zero_grad()

        start_time = time.time()
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            ddim_sampler = SequentialDDIM(timesteps = args.num_steps, 
                                  eta = args.eta, 
                                  cfg_scale = args.guidance_scale, 
                                  device = device,
                                  use_fp16 = False)

            sample = sequential_sampling(model, ddim_sampler, class_cond = class_cond, noise_vectors = noise_vectors)
            sample = decode_latent(vae, sample)

            losses = loss_fn(sample, ["null"] * sample.shape[0])
            loss = losses.mean()

            reward = -loss.item()
            if reward > best_reward:
                best_reward = reward
                best_sample = sample.detach()

            
            print("reward", reward, best_reward)
            print("scaler",  grad_scaler.get_scale())
            

         

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_([noise_vectors], 1.0)

            grad_scaler.step(optimizer)
            grad_scaler.update()

        end_time = time.time()
        print("time", end_time - start_time)

    
        img = to_img(sample)
        img = Image.fromarray(img)

        img.save(os.path.join(output_path, f"{i}_{reward}_{img_name}"))
        print("saved image")

if __name__ == "__main__":
    main()

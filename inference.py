"""
Inference for Composition-1k Dataset.

Run:
python inference.py \
    --config-dir path/to/config
    --checkpoint-dir path/to/checkpoint
    --inference-dir path/to/inference
    --data-dir path/to/data
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
import warnings
from data.dataset import DataGenerator
import time

warnings.filterwarnings("ignore")


# model and output
def matting_inference(
    config_dir="",
    checkpoint_dir="",
    inference_dir="",
    setname="",
):
    # initializing model
    torch.set_grad_enabled(False)
    cfg = LazyConfig.load(config_dir)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    DetectionCheckpointer(model).load(checkpoint_dir)
    model.eval()

    # initializing dataset
    test_dataset = DataGenerator(set_list=setname, phase="test", psm=cfg.hy_dict.psm, radius=cfg.hy_dict.radius)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

    # inferencing
    os.makedirs(inference_dir, exist_ok=True)

    start_time = time.time()
    for data in tqdm(test_loader):
        image_name = data["image_name"][0]
        H, W = data["hw"][0].item(), data["hw"][1].item()

        with torch.no_grad():
            pred = model(data)
            output = pred.flatten(0, 2) * 255
            output = cv2.resize(output.detach().cpu().numpy(), (W, H)).astype(np.uint8)
            output = F.to_pil_image(output).convert("RGB")
            output.save(join(inference_dir, image_name))
            torch.cuda.empty_cache()
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(test_loader)

    print(f"Start time: {start_time:.4f} seconds")
    print(f"End time: {end_time:.4f} seconds")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per iteration: {avg_time:.4f} seconds")


if __name__ == "__main__":
    # add argument we need:
    parser = default_argument_parser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--inference-dir", type=str, required=True)
    parser.add_argument("--setname", type=str, required=True)

    args = parser.parse_args()
    matting_inference(
        config_dir=args.config_dir,
        checkpoint_dir=args.checkpoint_dir,
        inference_dir=args.inference_dir,
        setname=args.setname,
    )


import json
import os

import numpy as np
import torch

from diffusers import DDIMScheduler
from transformers import T5EncoderModel
from transformer import Droplet3DTransformer3DModel
from vae import AutoencoderKLDroplet3D
from pipeline import Droplet3D_Pipeline_Inpaint
from PIL import Image
from utils import get_image_to_video_latent, save_videos_grid
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--width", type=int, default=512, help="The width of the output video"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="The height of the output video"
    )
    parser.add_argument(
        "--video_length", type=int, default=85, help="The length of the output video"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="The denoise steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The random seed"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="The guidance scale"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="Droplet3D-V1.0-weights",
        required=False,
        help="Path to the DropletVideo-5B model weights.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="This video features a cute cartoon panda astronaut. The panda is wearing a white spacesuit with a light and fun design style. The spacesuit is finely crafted, featuring all the details similar to real spacesuits, creating a strong sense of sci-fi and cartoon fusion. The panda's round face, with distinct black and white colors, conveys its classic appearance. Its eyebrows are black, and the large black patches around the eyes make it look even cuter and more lively. The panda's ears are round and full, peeking out from the sides of the astronaut helmet, enhancing the overall cartoon appeal. On the chest of the spacesuit, there is a blue panel, resembling a control instrument, surrounded by several pipes and buttons, adding some technological feel. The spacesuit also has red devices and design elements, with a badge-like decoration on the shoulders, adding more detail and layers to the outfit. The panda's gloves and boots are black, continuing the typical black and white color scheme, making it easy for children to love this design. The overall look presents a relaxed and friendly atmosphere, idealizing the image of a panda bravely exploring space. The video starts with a eye-level shot, first showing the panda astronaut's front view, allowing viewers to see its smile and the front details of the spacesuit. As the video continues to rotate, the side view becomes visible, making the panda's round ears and the backpack structure of the spacesuit more prominent. As the panda rotates on the screen, the back gradually appears, revealing the equipment on the back and the vest design. Finally, the panda completes a 360-degree rotation, allowing viewers to see the full-body view, and the video ends at this point. Each angle showcases the panda astronaut's different attractions, pleasant and imaginative.",
        required=False
    )

    parser.add_argument(
        "--view_align",
        action="store_true",
        help="whether to the canonical view alignment",
    )


    parser.add_argument(
        "--ref_img_dir",
        type=str,
        default="assets/Direct3D_panda_front_CW_R.png",
        required=False,
        help="the reference image dir",
    )
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    sample_width = args.width
    sample_height = args.height

    sample_size         = [sample_height, sample_width]
    video_length        = args.video_length
    weight_dtype            = torch.bfloat16
    validation_image_end    = None
    guidance_scale          = args.guidance_scale
    seed                    = args.seed
    num_inference_steps     = args.num_inference_steps

    model_name          = args.ckpt

    prompt = args.prompt
    ref_img_dir = args.ref_img_dir

    transformer = Droplet3DTransformer3DModel.from_pretrained_2d(
        model_name,
        subfolder="transformer",
    ).to(weight_dtype)


    vae = AutoencoderKLDroplet3D.from_pretrained(
        model_name,
        subfolder="vae"
    ).to(weight_dtype)

    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )

    scheduler = DDIMScheduler.from_pretrained(
        model_name,
        subfolder="scheduler"
    )

    pipeline = Droplet3D_Pipeline_Inpaint.from_pretrained(
        model_name,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype
    )

    if args.view_align:
        from diffusers import FluxKontextPipeline
        view_align_pipe = FluxKontextPipeline.from_pretrained(f"{model_name}/FLUX.1-Kontext-dev",
                                                   torch_dtype=torch.bfloat16)
        lora_weights_path = f"{model_name}/pytorch_lora_weights.safetensors"
        view_align_pipe.load_lora_weights(lora_weights_path)
        view_align_pipe.to("cuda")

    low_gpu_memory_mode = False
    if low_gpu_memory_mode:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.enable_model_cpu_offload()




    generator = torch.Generator(device="cuda").manual_seed(seed)
    prompt = prompt + " " + "The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
    negative_prompt         = "The video is low quality, blurry, distortion. "
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    os.makedirs("samples", exist_ok=True)
    if args.view_align:
        align_prompt = "Convert this image to the closest orthogonal view, such as front, left, or right."
        aligned_image = view_align_pipe(
            image=Image.open(ref_img_dir),
            prompt=align_prompt,
            guidance_scale=2.5,
            generator=torch.Generator().manual_seed(42),
            height=args.height,
            width=args.width,
            # max_area=512**2
        ).images[0]
        aligned_image = aligned_image.resize((args.width, args.height))
        aligned_image.save(os.path.join("samples", ref_img_dir.split("/")[-1][:-4] + "_aligned.jpg"))
        input_video, input_video_mask, clip_image = get_image_to_video_latent(aligned_image, validation_image_end,
                                                                              video_length=video_length,
                                                                              sample_size=sample_size)

    else:
        input_video, input_video_mask, clip_image = get_image_to_video_latent(ref_img_dir, validation_image_end, video_length=video_length, sample_size=sample_size)



    save_path = os.path.join("samples", ref_img_dir.split("/")[-1].replace(ref_img_dir.split("/")[-1][-4:], ".mp4"))
    with torch.no_grad():
        sample = pipeline(
            prompt,
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            max_sequence_length = 450,
            video        = input_video,
            mask_video   = input_video_mask,
        ).videos


        save_videos_grid(sample, save_path, fps=8)

    print("over")


if __name__ == "__main__":
    main()



















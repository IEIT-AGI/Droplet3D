<div align="center">

# Droplet3D

</div>

<br>

## ✈️ Introduction

**Droplet3D** is a project exploring high-order spatio-temporal consistency in image-to-MV generation. It is trained on Droplet3D-4M. The model supports canonical view alignment as well as the image-to-MV generation, and demonstrates potential for 3D consistency.

<br>

## 🚀 Installation
**Follow the steps below to set up the environment for our project.**

Our tested System Environment:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0

NVIDIA A100-SXM4-80GB
Driver Version: 550.144.03 


```


    
1. (Optional) Create a conda environment and activate it:
    
    ```bash
    conda create -n Droplet3D python=3.8
    conda activate Droplet3D
    ```
    
2. Install the required dependencies:
    
    ```bash
    cd Droplet3D_inference
    pip install -r requirements.txt
    ```
    
   We provide a `requirements.txt` file that contains all necessary dependencies for easy installation.



3. The Droplet3D-5B checkpoints can be downloaded from https://huggingface.co/DropletX/Droplet3D-5B.

   The distribution of internal model weights is as follows:
   
   The text_encoder as well as the tokenizer employs the google-t5 model weights(without training). The scheduler is the denoise strategy 
   during inference. The vae is the pixel-to-latent network for our project. The transformer contains our 5B transformer model weights. 
   Note: The FLUX.1-Kontext-dev folder contains the original weights of FLUX.1-Kontext-dev model.  The pytorch_lora_weights.safetensors file is 
   the lora weights file for the view alignment in the paper.

    ```
    Droplet3D-5B/
    ├── configuration.json
    ├── LICENSE
    ├── model_index.json
    ├── pytorch_lora_weights.safetensors
    ├── README.md
    ├── scheduler
    │     └── scheduler_config.json
    ├── text_encoder
    │     ├── config.json
    │     ├── model-00001-of-00002.safetensors
    │     ├── model-00002-of-00002.safetensors
    │     └── model.safetensors.index.json
    ├── tokenizer
    │     ├── added_tokens.json
    │     ├── special_tokens_map.json
    │     ├── spiece.model
    │     └── tokenizer_config.json
    ├── FLUX.1-Kontext-dev
    ├── transformer
    │     ├── config.json
    │     └── diffusion_pytorch_model.safetensors
    └── vae
          ├── config.json
          └── diffusion_pytorch_model.safetensors
    ```   


#### Notation:
   
   All the model weights are stored in safetensors. Satetensors is a file format designed for stroing tensor data, aiming to provide efficient
   and secure read and write operations. It is commonly used to store weights and parameters in machine learning models. Below are methods for reading
   safetensors. You can check the model_weights from the state_dict variable.
   
   ```
   from safetensors.torch import load_file
   state_dict = load_file(file_path)
   ```


<br>

## ⚡ Usage
Once the installation is complete, you can run the demo using the following command:

```bash
python inference.py --ckpt Droplet3D-5B --ref_img_dir your_path_to_ref_img --prompt yout_text_input --view_align
```

#### Example:
```bash
python inference.py --ckpt Droplet3D-5B --ref_img_dir assets/1.jpg --prompt "This video features a cute cartoon panda astronaut. The panda wears a white spacesuit designed in a lighthearted and playful style.
The spacesuit is meticulously crafted, featuring all the details reminiscent of real-life spacesuits, blending a strong sense of
science fiction with cartoon charm. The panda's round face is rendered in bold black-and-white colors, capturing its classic
appearance. Its eyebrows are black, and the large black patterns around its eyes make it look even cuter and more lively. The
panda's ears are round and full, peeking out from the sides of the spacesuit helmet, enhancing the overall cartoon appeal. The
chest of the spacesuit features a blue panel, resembling a control panel for certain functions, surrounded by several tubes and
buttons, adding a touch of technological sophistication. The spacesuit is adorned with red devices and design elements, including
badge-like decorations on the shoulders, adding depth to the overall outfit's details. The panda's gloves and boots are black,
continuing its classic black-and-white color scheme, making it easy for children to fall in love with this design. The overall design
exudes a relaxed and friendly vibe, idealizing the image of a panda bravely exploring space.
The video begins with an eye-level shot, first showcasing the front of the panda astronaut. From this angle, its smiling face and
the detailed design of the entire spacesuit are visible. As the video continues to rotate, a side view is revealed, making the panda’s
round ears and the structure of the spacesuit’s backpack more prominent. As the panda turns on screen, its back gradually comes
into view, displaying the equipment and the design of its vest. Finally, the panda completes a full 360-degree rotation, allowing
the viewer to see the complete, full-body perspective before the video ends."
```

### Command Line Arguments

#### 1. required arguments
- `--ckpt`: Path to the model weights.
- `--ref_img_dir`: The input condition img path
- `--view_align`: whether to align the inputview
- `--prompt`: The input text


#### 2. Other arguments
- `--width`: The width of the generated video
- `--height`: The height of the generated video
- `--video_length`: The frame num of the generated video
- `--num_inference_steps`: The denoise step for inference. Normally, the quality of the generated video will be better 
                           if the value is higher but with higher computation cost. Normally, we set it to 50.
- `--seed`: The random seed for the inference, different seeds will generate different results.
- `--guidance_scale`: The guidance scale of the denoise process. The value determines the relationship between the input 
                      prompt and the generated video. The higher value, the more relative. 







<br>



<br>

## 🙏 Credits
This project leverages the following open-source frameworks:
+ [**CogVideoX-Fun**](https://github.com/aigc-apps/CogVideoX-Fun)
+ [**CogVideoX**](https://github.com/THUDM/CogVideo)

<br>

## ☎️ Contact us
If you have any questions, comments, or suggestions, please contact us at [zrzsgsg@gmail.com](mailto:zrzsgsg@gmail.com).

<br>

## 📄 License
This project is released under the [Apache 2.0 license](resources/LICENSE).


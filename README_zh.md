<div align="center">

# Droplet3D

</div>

<br>

## ✈️ Introduction

**Droplet3D**通过利用视频作为辅助监督信号来缓解3D生成中的数据稀缺问题。视频提供多视图信息，带来空间一致性；其丰富的语义有助于生成内容更好地与文本提示对齐。该工作介绍了 Droplet3D-4M，这是一个具有多视图标注的大规模视频数据集，以及 Droplet3D，它是一个支持图像和密集文本输入的生成模型。大量实验证明，该方法能产生空间上更一致、语义上更合理的结果，并且具有场景级应用的潜力，凸显了视频先验在3D创作中的有益作用。

<br>

<p align="center">
  English | <a href="README_zh.md">简体中文</a>
</p>


## 🚀 安装
**请按照以下步骤为我们的项目设置环境。**

我们的测试系统环境如下:

```
nvcc: NVIDIA (R) CUDA 编译器驱动
版权所有 (c) 2005-2022 NVIDIA Corporation
在 2022 年 9 月 21 日编译
Cuda 编译工具包，版本 11.8
构建 cuda_11.8.r11.8/compiler.31833905_0

NVIDIA A100-SXM4-80GB
驱动版本: 550.144.03



```


    
1. （可选）创建并激活一个 conda 环境:
    
    ```bash
    conda create -n Droplet3D python=3.8
    conda activate Droplet3D
    ```
    
2. 安装必需的依赖项：
    
    ```bash
    cd Droplet3D_inference
    pip install -r requirements.txt
    ```
    
   我们提供了一个 requirements.txt 文件，包含用于简便安装的所有依赖项。


3. Droplet3D-5B 的权重可从 https://huggingface.co/DropletX/Droplet3D-5B 下载

    内部模型权重分布如下：
    
    
    text_encoder 以及 tokenizer 使用 google-t5 的未训练权重。调度器（scheduler）在推断阶段使用去噪策略。vae 是将像素映射到潜在表示的网络。transformer 含有我们的 5B         Transformer 模型权重。
    注：FLUX.1-Kontext-dev 文件夹包含原始的 FLUX.1-Kontext-dev 模型权重。pytorch_lora_weights.safetensors 是用于视图对齐的 LoRA 权重文件。



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


#### 说明:
   
    所有模型权重均存储为 safetensors。safetensors 是一种专门用于高效且安全地读写张量数据的文件格式，常用于存储权重与参数。以下是读取 safetensors 的方法。你可以通过             state_dict 变量查看模型权重。
   
   ```
   from safetensors.torch import load_file
   state_dict = load_file(file_path)
   ```


<br>

## ⚡ 使用
安装完成后，使用以下命令运行演示（demo）:

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

### 命令行参数

#### 1. 必须参数
- `--ckpt`: 模型权重路径
- `--ref_img_dir`: 输入的条件图像路径
- `--view_align`: 是否执行视角对齐
- `--prompt`: 输入文本


#### 2. 其它参数
- `--width`: 生成视频的宽度
- `--height`: 生成视频的高度
- `--video_length`: 生成视频的帧数
- `--num_inference_steps`: 推断的去噪步数。步数越大通常质量越好，但计算成本也越高。通常设为 50。
- `--seed`: 推断的随机种子，不同种子会产生不同结果。
- `--guidance_scale`:去噪过程中引导强度，数值越高越贴合输入提示文本。







<br>



<br>

## 🙏 致谢
本项目借助以下开源框架：
+ [**CogVideoX-Fun**](https://github.com/aigc-apps/CogVideoX-Fun)
+ [**CogVideoX**](https://github.com/THUDM/CogVideo)
+ [**DropletVideo**](https://github.com/IEIT-AGI/DropletVideo)

<br>

## 引用

🌟 如本文工作对你有帮助，请给我们点星并在论文引用中标注。


```
@article{li2025droplet3d,
      title={Droplet3D: Commonsense Priors from Videos Facilitate 3D Generation},
      author={Li, Xiaochuan and Du, Guoguang and Zhang, Runze and Jin, Liang and Jia, Qi and Lu, Lihua and Guo, Zhenhua and Zhao, Yaqian and Liu, Haiyang and Wang, Tianqi and Li, Changsheng and Gong, Xiaoli and Li, Rengang and Fan, Baoyu},
      journal={arXiv preprint arXiv:2508.20470},
      year={2025}
    }
```


## ☎️ Contact us
If you have any questions, comments, or suggestions, please contact us at [zrzsgsg@gmail.com](mailto:zrzsgsg@gmail.com).

<br>

## 📄 License
This project is released under the [Apache 2.0 license](resources/LICENSE).


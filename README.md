# ⚠️ Notice

This repository is a **fork** of the original [CCEdit](https://github.com/RuoyuFeng/CCEdit?tab=readme-ov-file) project.  
It includes additional implementations and modifications by **Raito Sugihara** for research purposes (MSc Dissertation, University of Surrey, 2025).  

### Additional Changes
- Added bash scripts for **inference**
- Enabled style transfer by integrating **pre-trained base models** through the bash scripts

### Usage Notes
Using only the instructions from the original README may lead to inference failures.  
To resolve this, it is necessary to provide a **pre-trained base model** via the `--basemodel_path` argument.  
The models recommended in the original paper were employed, and the following pre-trained models were used in this fork:

- **revAnimated v2**  
- **ToonYou alpha3**  
- **hellomecha V12**  
- **Counterfeit V3 / PixelArtRedmond 15V**

All models were obtained from **Hugging Face** and **Civitai**.  
⚠️ Due to their large file sizes, these pre-trained base models are **not included in this repository** and must be downloaded separately.

→ For further installation and usage instructions, please refer to the **Original README** provided below.  
This fork mainly highlights the additional implementations listed above.

---

# Original README


### <div align="center"> CCEdit: Creative and Controllable Video Editing via Diffusion Models<div> 
### <div align="center"> CVPR 2024 <div> 


<div align="center">
Ruoyu Feng,
Wenming Weng,
Yanhui Wang,
Yuhui Yuan,
Jianmin Bao,
Chong Luo,
Zhibo Chen,
Baining Guo
</div>

<br>

<div align="center">
  <a href="https://ruoyufeng.github.io/CCEdit.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://huggingface.co/datasets/RuoyuFeng/BalanceCC"><img src="https://img.shields.io/static/v1?label=BalanceCC BenchMark&message=HF&color=yellow"></a> &ensp;
  <a href="https://arxiv.org/pdf/2309.16496.pdf"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:CCEdit&color=red&logo=arxiv"></a> &ensp;
</div>

<table class="center">
    <tr>
    <td><img src="assets/makeup.gif"></td>
    <td><img src="assets/makeup1-magicReal.gif"></td>
    </tr>
</table>

## 🔥 Update
- 🔥 Mar. 27, 2024. [BalanceCC Benchmark](https://huggingface.co/datasets/RuoyuFeng/BalanceCC) is released! BalanceCC benchmark contains 100 videos with varied attributes, designed to offer a comprehensive platform for evaluating generative video editing, focusing on both controllability and creativity.

## Installation
```
# env
conda create -n ccedit python=3.9.17
conda activate ccedit
pip install -r requirements.txt
# pip install -r requirements_pt2.txt
# pip install torch==2.0.1 torchaudio==2.0.2 torchdata==0.6.1 torchmetrics==1.0.0 torchvision==0.15.2
pip install basicsr==1.4.2 wandb loralib av decord timm==0.6.7
pip install moviepy imageio==2.6.0 scikit-image==0.20.0 scipy==1.9.1 diffusers==0.17.1 transformers==4.27.3
pip install accelerate==0.20.3 ujson

git clone https://github.com/lllyasviel/ControlNet-v1-1-nightly src/controlnet11
git clone https://github.com/MichalGeyer/pnp-diffusers src/pnp-diffusers
```

# Download models
download models from https://huggingface.co/RuoyuFeng/CCEdit and put them in ./models

<!-- ## Inference and training examples -->
## Inference
### Text-Video-to-Video
```bash
python scripts/sampling/sampling_tv2v.py   --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml   --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt  --H 512 --W 768 --original_fps 18 --target_fps 6 --num_keyframes 17 --batch_size 1 --num_samples 2   --sample_steps 30 --sampler_name DPMPP2SAncestralSampler  --cfg_scale 7.5   --prompt 'a bear is walking.' --video_path assets/Samples/davis/bear   --add_prompt 'Van Gogh style'   --save_path outputs/tv2v/bear-VanGogh   --disable_check_repeat
```

### Text-Video-Image-to-Video
Specifiy the edited center frame.
```bash
python scripts/sampling/sampling_tv2v_ref.py \
    --seed 201574 \
    --config_path configs/inference_ccedit/keyframe_ref_cp_no2ndca_add_cfca_depthzoe.yaml \
    --ckpt_path models/tvi2v-no2ndca-depthmidas.ckpt \
    --H 512 --W 768 --original_fps 18 --target_fps 6 --num_keyframes 17 --batch_size 1 --num_samples 2 \
    --sample_steps 50 --sampler_name DPMPP2SAncestralSampler --cfg_scale 7 \
    --prompt 'A person walks on the grass, the Milky Way is in the sky, night' \
    --add_prompt 'masterpiece, best quality,' \
    --video_path assets/Samples/tshirtman.mp4 \
    --reference_path assets/Samples/tshirtman-milkyway.png \
    --save_path outputs/tvi2v/tshirtman-MilkyWay \
    --disable_check_repeat \
    --prior_coefficient_x 0.03 \
    --prior_type ref
```

Automatic edit the center frame via [pnp-diffusers](https://github.com/MichalGeyer/pnp-diffusers)
Note that the performance of this pipeline heavily depends on the quality of the automatic editing result. So try to use more powerful automatic editing methods to edit the center frame. Or we recommond combine CCEdit with other powerfull AI editing tools, such as Stable-Diffusion WebUI, comfyui, etc.
```bash
# python preprocess.py --data_path <path_to_guidance_image> --inversion_prompt <inversion_prompt>
python src/pnp-diffusers/preprocess.py --data_path assets/Samples/tshirtman-milkyway.png --inversion_prompt 'a man walks in the filed'
# modify the config file (config_pnp.yaml) to use the processed image
# python pnp.py --config_path <pnp_config_path>
python src/pnp-diffusers/pnp.py --config_path config_pnp.yaml
python scripts/sampling/sampling_tv2v_ref.py \
    --seed 201574 \
    --config_path configs/inference_ccedit/keyframe_ref_cp_no2ndca_add_cfca_depthzoe.yaml \
    --ckpt_path models/tvi2v-no2ndca-depthmidas.ckpt \
    --H 512 --W 768 --original_fps 18 --target_fps 6 --num_keyframes 17 --batch_size 1 --num_samples 2 \
    --sample_steps 50 --sampler_name DPMPP2SAncestralSampler --cfg_scale 7 \
    --prompt 'A person walks on the grass, the Milky Way is in the sky, night' \
    --add_prompt 'masterpiece, best quality,' \
    --video_path assets/Samples/tshirtman.mp4 \
    --reference_path "PNP-results/tshirtman-milkyway/output-a man walks in the filed, milky way.png" \
    --save_path outputs/tvi2v/tshirtman-MilkyWay \
    --disable_check_repeat \
    --prior_coefficient_x 0.03 \
    --prior_type ref
```

You can use the following pipeline to automatically extract the center frame, conduct editing via pnp-diffusers and then conduct video editing via tvi2v.
```bash
python scripts/sampling/pnp_generate_config.py \
    --p_config config_pnp_auto.yaml \
    --output_path "outputs/automatic_ref_editing/image" \
    --image_path "outputs/centerframe/tshirtman.png" \
    --latents_path "latents_forward" \
    --prompt "a man walks on the beach" 
python scripts/tools/extract_centerframe.py \
    --p_video assets/Samples/tshirtman.mp4 \
    --p_save outputs/centerframe/tshirtman.png \
    --orifps 18 \
    --targetfps 6 \
    --n_keyframes 17 \
    --length_long 512 \
    --length_short 512
python src/pnp-diffusers/preprocess.py --data_path outputs/centerframe/tshirtman.png --inversion_prompt 'a man walks in the filed'
python src/pnp-diffusers/pnp.py --config_path config_pnp_auto.yaml
python scripts/sampling/sampling_tv2v_ref.py \
    --seed 201574 \
    --config_path configs/inference_ccedit/keyframe_ref_cp_no2ndca_add_cfca_depthzoe.yaml \
    --ckpt_path models/tvi2v-no2ndca-depthmidas.ckpt \
    --H 512 --W 768 --original_fps 18 --target_fps 6 --num_keyframes 17 --batch_size 1 --num_samples 2 \
    --sample_steps 50 --sampler_name DPMPP2SAncestralSampler --cfg_scale 7 \
    --prompt 'A man walks on the beach' \
    --add_prompt 'masterpiece, best quality,' \
    --video_path assets/Samples/tshirtman.mp4 \
    --reference_path "outputs/automatic_ref_editing/image/output-a man walks on the beach.png" \
    --save_path outputs/tvi2v/tshirtman-Beach \
    --disable_check_repeat \
    --prior_coefficient_x 0.03 \
    --prior_type ref
```

## Train example
```bash
python main.py -b configs/example_training/sd_1_5_controlldm-test-ruoyu-tv2v-depthmidas.yaml --wandb False
```

## BibTeX
If you find this work useful for your research, please cite us:

```
@article{feng2023ccedit,
  title={CCEdit: Creative and Controllable Video Editing via Diffusion Models},
  author={Feng, Ruoyu and Weng, Wenming and Wang, Yanhui and Yuan, Yuhui and Bao, Jianmin and Luo, Chong and Chen, Zhibo and Guo, Baining},
  journal={arXiv preprint arXiv:2309.16496},
  year={2023}
}
```

## Conact Us
**Ruoyu Feng**: [ustcfry@mail.ustc.edu.cn](ustcfry@mail.ustc.edu.cn)  


## Acknowledgements
The source videos in this repository come from our own collections and downloads from Pexels. If anyone feels that a particular piece of content is used inappropriately, please feel free to contact me, and I will remove it immediately.

Thanks to model contributers of [CivitAI](https://civitai.com/) and [RunwayML](https://runwayml.com/).

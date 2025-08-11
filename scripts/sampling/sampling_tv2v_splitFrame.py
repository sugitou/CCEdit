import argparse
import json
import os
import random

import torch
import torchvision
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from safetensors import safe_open
from torch import autocast

from scripts.sampling.util import (
    chunk,
    convert_load_lora,
    create_model,
    init_sampling,
    load_video_keyframes,
    model_load_ckpt,
    perform_save_locally_video,
)
from sgm.util import append_dims


def _linear_blend_time(prev: torch.Tensor, curr: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    prev, curr: (B, C, T, H, W) を時間方向で結合。重なり T=overlap を線形ブレンド。
    戻り値: (B, C, T_prev+T_curr-overlap, H, W)
    """
    if overlap <= 0:
        return torch.cat([prev, curr], dim=2)
    assert prev.shape[:2] == curr.shape[:2] and prev.shape[3:] == curr.shape[3:], "shape mismatch"
    Tp, Tc = prev.shape[2], curr.shape[2]
    assert Tp >= overlap and Tc >= overlap, "overlap longer than clip"

    w = torch.linspace(0, 1, steps=overlap, device=prev.device, dtype=prev.dtype)
    w = w.view(1, 1, overlap, 1, 1)  # (1,1,T,1,1)

    prev_body  = prev[:, :, :-overlap]
    prev_tail  = prev[:, :, -overlap:]
    curr_head  = curr[:, :, :overlap]
    curr_body  = curr[:, :, overlap:]

    blended = prev_tail * (1 - w) + curr_head * w
    return torch.cat([prev_body, blended, curr_body], dim=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--use_default", action="store_true", help="use default ckpt at first"
    )
    parser.add_argument(
        "--basemodel_path",
        type=str,
        default="",
        help="load a new base model instead of original sd-1.5",
    )
    parser.add_argument("--basemodel_listpath", type=str, default="")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--vae_path", type=str, default="")
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
    )
    parser.add_argument("--prompt_listpath", type=str, default="")
    parser.add_argument("--video_listpath", type=str, default="")
    parser.add_argument(
        "--videos_directory",
        type=str,
        default="",
        help="directory containing videos to be processed",
    )
    parser.add_argument(
        '--json_path',
        type=str,
        default='',
        help='path to json file containing video paths and captions'
    )
    parser.add_argument(
        '--videos_root',
        type=str,
        default='',
        help='path to the root of videos'
    )
    parser.add_argument("--save_path", type=str, default="outputs/demo/tv2v")
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=384)
    parser.add_argument("--detect_ratio", type=float, default=1.0)
    parser.add_argument("--original_fps", type=int, default=20)
    parser.add_argument("--target_fps", type=int, default=3)
    parser.add_argument("--num_keyframes", type=int, default=9)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="ugly, low quality")
    parser.add_argument("--add_prompt", type=str, default="masterpiece, high quality")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler_name", type=str, default="EulerEDMSampler")
    parser.add_argument(
        "--discretization_name", type=str, default="LegacyDDPMDiscretization"
    )
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--prior_coefficient_x", type=float, default=0.0)
    parser.add_argument("--prior_coefficient_noise", type=float, default=1.0)
    parser.add_argument("--sdedit_denoise_strength", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--disable_check_repeat', action='store_true', help='disable check repeat')
    parser.add_argument('--lora_strength', type=float, default=0.8)
    parser.add_argument('--save_type', type=str, default='mp4', choices=['gif', 'mp4'])
    parser.add_argument('--inpainting_mode', action='store_true', help='inpainting mode')
    # Add
    # argparse に追加
    parser.add_argument(
        "--window_size", type=int, default=0, help="Temporal window length for sliding-window generation. 0 disables it."
    )
    parser.add_argument(
        "--window_overlap", type=int, default=0, help="Temporal overlap between consecutive windows."
    )

    args = parser.parse_args()

    seed = args.seed
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed_everything(seed)

    # initialize the model
    model = create_model(config_path=args.config_path).to("cuda")
    ckpt_path = args.ckpt_path
    print("--> load ckpt from: ", ckpt_path)
    model = model_load_ckpt(model, path=ckpt_path)
    model.eval()

    # load the prompts and video_paths
    video_save_paths = []
    assert not (args.prompt_listpath and args.videos_directory), (
        "Only one of prompt_listpath and videos_directory can be provided, "
        "but got prompt_listpath: {}, videos_directory: {}".format(
            args.prompt_listpath, args.videos_directory
        )
    )
    if args.prompt_listpath:
        with open(args.prompt_listpath, "r") as f:
            prompts = f.readlines()
        prompts = [p.strip() for p in prompts]
        # load paths of cond_img
        assert args.video_listpath, (
            "video_listpath must be provided when prompt_listpath is provided, "
            "but got video_listpath: {}".format(args.video_listpath)
        )
        with open(args.video_listpath, "r") as f:
            video_paths = f.readlines()
        video_paths = [p.strip() for p in video_paths]
    elif args.videos_directory:
        prompts = []
        video_paths = []
        for video_name in os.listdir(args.videos_directory):
            video_path = os.path.join(args.videos_directory, video_name)
            if os.path.isdir(video_path):
                prompts.append(video_name)
                video_paths.append(video_path)
    elif args.json_path:
        assert args.videos_root != '', 'videos_root must be provided when json_path is provided'
        with open(args.json_path, 'r') as f:
            json_dict = json.load(f)
        prompts = []
        video_paths = []
        for item in json_dict:
            video_path = os.path.join(args.videos_root, item["Video Type"], item["Video Name"] + '.mp4')
            
            for edit in item['Editing']:
                video_paths.append(video_path)
                prompts.append(edit["Target Prompt"])
                video_save_paths.append(
                    os.path.join(args.save_path, item["Video Type"], item["Video Name"], edit["Target Prompt"])
                )
    else:
        assert args.prompt and args.video_path, (
            "prompt and video_path must be provided when prompt_listpath and videos_directory are not provided, "
            "but got prompt: {}, video_path: {}".format(args.prompt, args.video_path)
        )
        prompts = [args.prompt]
        video_paths = [args.video_path]

    assert len(prompts) == len(
        video_paths
    ), "The number of prompts and video_paths must be the same, and you provided {} prompts and {} video_paths".format(
        len(prompts), len(video_paths)
    )
    num_samples = args.num_samples
    batch_size = args.batch_size

    print("\nNumber of prompts: {}".format(len(prompts)))
    print("Generate {} samples for each prompt".format(num_samples))

    prompts = [item for item in prompts for _ in range(num_samples)]
    video_paths = [item for item in video_paths for _ in range(num_samples)]

    prompts_chunk = list(chunk(prompts, batch_size))
    video_paths_chunk = list(chunk(video_paths, batch_size))
    del prompts
    del video_paths

    # load paths of basemodel if provided
    assert not (args.basemodel_path and args.basemodel_listpath), (
        "Only one of basemodel_path and basemodel_listpath can be provided, "
        "but got basemodel_path: {}, basemodel_listpath: {}".format(
            args.basemodel_path, args.basemodel_listpath
        )
    )
    basemodel_paths = []
    if args.basemodel_listpath:
        with open(args.basemodel_listpath, "r") as f:
            basemodel_paths = f.readlines()
        basemodel_paths = [p.strip() for p in basemodel_paths]
    if args.basemodel_path:
        basemodel_paths = [args.basemodel_path]
    if args.use_default:
        basemodel_paths = ["default"] + basemodel_paths
    if len(basemodel_paths) == 0:
        basemodel_paths = ["default"]

    for basemodel_idx, basemodel_path in enumerate(basemodel_paths):
        print("-> base model idx: ", basemodel_idx)
        print("-> base model path: ", basemodel_path)

        if basemodel_path == "default":
            pass
        elif basemodel_path:
            print("--> load a new base model from {}".format(basemodel_path))
            model = model_load_ckpt(model, basemodel_path, True)

        if args.lora_path:
            print("--> load a new LoRA model from {}".format(args.lora_path))
            sd_state_dict = model.state_dict()
            lora_path = args.lora_path

            if lora_path.endswith(".safetensors"):
                lora_state_dict = {}

                # with safe_open(lora_path, framework="pt", device='cpu') as f:
                with safe_open(lora_path, framework="pt", device=0) as f:
                    for key in f.keys():
                        lora_state_dict[key] = f.get_tensor(key)

                is_lora = all("lora" in k for k in lora_state_dict.keys())
                if not is_lora:
                    raise ValueError(
                        f"The model you provided in [{lora_path}] is not a LoRA model. "
                    )
            else:
                raise NotImplementedError
            sd_state_dict = convert_load_lora(
                sd_state_dict, lora_state_dict, alpha=args.lora_strength
            )  #
            model.load_state_dict(sd_state_dict)

        # TODO: the logic here is not elegant.
        if args.vae_path:
            vae_path = args.vae_path
            print("--> load a new VAE model from {}".format(vae_path))

            if vae_path.endswith(".pt"):
                vae_state_dict = torch.load(vae_path, map_location="cpu")["state_dict"]
                msg = model.first_stage_model.load_state_dict(
                    vae_state_dict, strict=False
                )
            elif vae_path.endswith(".safetensors"):
                vae_state_dict = {}

                # with safe_open(vae_path, framework="pt", device='cpu') as f:
                with safe_open(vae_path, framework="pt", device=0) as f:
                    for key in f.keys():
                        vae_state_dict[key] = f.get_tensor(key)

                msg = model.first_stage_model.load_state_dict(
                    vae_state_dict, strict=False
                )
            else:
                raise ValueError("Cannot load vae model from {}".format(vae_path))

            print("msg of loading vae: ", msg)

        if os.path.exists(
            os.path.join(
                args.save_path,
                basemodel_path.split("/")[-1].split(".")[0],
                "log_info.json",
            )
        ):
            with open(
                os.path.join(
                    args.save_path,
                    basemodel_path.split("/")[-1].split(".")[0],
                    "log_info.json",
                ),
                "r",
            ) as f:
                log_info = json.load(f)
        else:
            log_info = {
                "basemodel_path": basemodel_path,
                "lora_path": args.lora_path,
                "vae_path": args.vae_path,
                "video_paths": [],
                "keyframes_paths": [],
            }

        num_keyframes = args.num_keyframes

        for idx, (prompts, video_paths) in enumerate(
            zip(prompts_chunk, video_paths_chunk)
        ):
            # if idx == 2: # ! DEBUG
            #     break
            if not args.disable_check_repeat:
                while video_paths[0] in log_info["video_paths"]:
                    print(f"video [{video_paths[0]}] has been processed, skip it.")
                    prompts_list, video_paths_list = list(prompts), list(video_paths)
                    prompts_list.pop(0)
                    video_paths_list.pop(0)
                    prompts, video_paths = tuple(prompts_list), tuple(video_paths_list)
                    del prompts_list, video_paths_list
                    if len(prompts) == 0:
                        break
                if len(video_paths) == 0:
                    continue

            bs = min(len(prompts), batch_size)
            print(f"\nProgress: {idx} / {len(prompts_chunk)}. ")
            H, W = args.H, args.W
            keyframes_list = []
            print("load video ...")
            try:
                for video_path in video_paths:
                    keyframes = load_video_keyframes(
                        video_path,
                        args.original_fps,
                        args.target_fps,
                        num_keyframes,
                        (H, W),
                    )
                    keyframes = keyframes.unsqueeze(0)  # B T C H W
                    keyframes = rearrange(keyframes, "b t c h w -> b c t h w").to(
                        model.device
                    )
                    keyframes_list.append(keyframes)

                    # DEBUG: Save keyframes as images
                    # save_dir = "debug_keyframes"
                    # os.makedirs(save_dir, exist_ok=True)
                    # frames_to_save = rearrange(keyframes[0], 'c t h w -> t c h w')  # (T, C, H, W)
                    # for i, frame in enumerate(frames_to_save):
                    #     torchvision.utils.save_image(frame, f"{save_dir}/frame_{i}.png")

            except:
                print(f"Error when loading video from  {video_paths}")
                continue
            print("load video done ...")
            keyframes = torch.cat(keyframes_list, dim=0)
            control_hint = keyframes

            # DEBUG: Check statistics of control_hint
            print("control_hint shape:", control_hint.shape)
            print("control_hint mean:", control_hint.mean().item())
            print("control_hint std:", control_hint.std().item())

            # ---------- ここから置き換え開始 ----------

            # 以降で使う共通
            negative_prompt = args.negative_prompt
            bs = min(len(prompts), batch_size)
            H, W = args.H, args.W
            sampling_kwargs = {}  # usually empty

            # プロンプト前処理を先に固定
            txts = list(prompts)
            if args.add_prompt:
                txts = [args.add_prompt + ", " + each for each in txts]

            def _sample_one_window(control_hint_win: torch.Tensor) -> torch.Tensor:
                """
                control_hint_win: (B, C, Tw, H, W)
                戻り値: decode後の画素空間 (B, C, Tw, H, W), [0,1]
                """
                Tw = control_hint_win.shape[2]

                # per-window batch / cond
                batch_win = {
                    "txt": txts,
                    "control_hint": control_hint_win,
                }
                batch_uc_win = {
                    "txt": [negative_prompt for _ in range(bs)],
                    # 事前学習重みに合わせ uc 側も同じ control_hint を使う仕様
                    "control_hint": control_hint_win.clone(),
                }

                c_win, uc_win = model.conditioner.get_unconditional_conditioning(
                    batch_c=batch_win, batch_uc=batch_uc_win,
                )

                for k in c_win:
                    if isinstance(c_win[k], torch.Tensor):
                        c_win[k], uc_win[k] = map(lambda y: y[k][:bs].to(model.device), (c_win, uc_win))

                shape_win = (4, Tw, H // 8, W // 8)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    randn = torch.randn(bs, *shape_win, device=model.device)

                    if args.sdedit_denoise_strength == 0.0:
                        def denoiser(x, sigma, cdict):
                            return model.denoiser(model.model, x, sigma, cdict, **sampling_kwargs)

                        if args.prior_coefficient_x != 0.0:
                            # ウィンドウ分だけのpriorを使用
                            prior = model.encode_first_stage(control_hint_win)
                            randn = args.prior_coefficient_x * prior + args.prior_coefficient_noise * randn

                        sampler = init_sampling(
                            sample_steps=args.sample_steps,
                            sampler_name=args.sampler_name,
                            discretization_name=args.discretization_name,
                            guider_config_target="sgm.modules.diffusionmodules.guiders.VanillaCFGTV2V",
                            cfg_scale=args.cfg_scale,
                        )
                        sampler.verbose = True
                        samples_lat = sampler(denoiser, randn, c_win, uc=uc_win)

                    else:
                        # sdedit（img2img）分岐
                        denoise_strength = args.sdedit_denoise_strength
                        assert args.prior_coefficient_x == 0, "Use either sdedit or prior_coefficient_x, not both."
                        sampler = init_sampling(
                            sample_steps=args.sample_steps,
                            sampler_name=args.sampler_name,
                            discretization_name=args.discretization_name,
                            guider_config_target="sgm.modules.diffusionmodules.guiders.VanillaCFGTV2V",
                            cfg_scale=args.cfg_scale,
                            img2img_strength=denoise_strength,
                        )
                        sampler.verbose = True
                        z = model.encode_first_stage(control_hint_win)
                        noise = torch.randn_like(z)
                        sigmas = sampler.discretization(sampler.num_steps).to(z.device)
                        sigma = sigmas[0]
                        noised_z = (z + noise * append_dims(sigma, z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)

                        def denoiser(x, sigma, cdict):
                            return model.denoiser(model.model, x, sigma, cdict)

                        samples_lat = sampler(denoiser, noised_z, cond=c_win, uc=uc_win)

                    samples_img = model.decode_first_stage(samples_lat)
                    # samples_img = (torch.clamp(samples_img, -1.0, 1.0) + 1.0) / 2.0  # [0,1]
                    # クリーニング
                    del samples_lat, randn
                    if args.sdedit_denoise_strength > 0.0:
                        del z, noise, sigmas, sigma, noised_z
                    torch.cuda.empty_cache()
                    return samples_img

            # ==== 窓あり/なし分岐 ====
            T_total = control_hint.shape[2]
            use_window = (args.window_size > 0) and (args.window_size < T_total)
            if not use_window:
                # 既存フロー（全フレーム一括）
                samples = _sample_one_window(control_hint)
            else:
                win = int(args.window_size)
                ov  = int(args.window_overlap)
                assert win > 0 and win <= T_total, "window_size must be in (0, T]"
                assert 0 <= ov < win, "window_overlap must be in [0, window_size)"
                stride = win - ov

                assembled = None
                start = 0
                while start < T_total:
                    end = min(start + win, T_total)
                    ctrl_win = control_hint[:, :, start:end, :, :]
                    samp_win = _sample_one_window(ctrl_win)

                    if assembled is None:
                        assembled = samp_win
                    else:
                        # 時間方向ブレンド結合
                        assembled = _linear_blend_time(assembled, samp_win, overlap=min(ov, assembled.shape[2], samp_win.shape[2]))

                    # 後処理
                    del ctrl_win, samp_win
                    torch.cuda.empty_cache()
                    start += stride

                samples = assembled
                del assembled
                torch.cuda.empty_cache()

            # ---------- ここまで置き換え ----------


            # save the results
            keyframes = (torch.clamp(keyframes, -1.0, 1.0) + 1.0) / 2.0
            samples = (torch.clamp(samples, -1.0, 1.0) + 1.0) / 2.0
            control_hint = (torch.clamp(control_hint, -1.0, 1.0) + 1.0) / 2.0
            if video_save_paths == []:
                save_path = args.save_path
                save_path = os.path.join(
                    save_path, basemodel_path.split("/")[-1].split(".")[0]
                )
            else:
                save_path = video_save_paths[idx]

            perform_save_locally_video(
                os.path.join(save_path, "original"), 
                keyframes, 
                args.target_fps, 
                args.save_type,
                save_grid=False
            )

            keyframes_paths = perform_save_locally_video(
                os.path.join(save_path, "result"),
                samples,
                args.target_fps,
                args.save_type,
                return_savepaths=True,
                save_grid=False
            )
            perform_save_locally_video(
                os.path.join(save_path, "control_hint"),
                control_hint,
                args.target_fps,
                args.save_type,
                save_grid=False
            )
            print("Saved samples to {}. Enjoy.".format(save_path))

            # save video paths
            log_info["video_paths"] += video_paths
            log_info["keyframes_paths"] += keyframes_paths

            # save log info
            with open(os.path.join(save_path, "log_info.json"), "w") as f:
                json.dump(log_info, f, indent=4)
            
            # ---------- Insert this block ----------
            # Explicitly delete large tensors and free GPU memory
            to_delete = [
                "keyframes", "control_hint", "c", "uc", "randn", "samples",
                "z", "noise", "sigmas", "noised_z", "mask", "prior", "noised_z",
                "batch", "batch_uc"
            ]
            for name in to_delete:
                if name in locals():
                    del locals()[name]
            torch.cuda.empty_cache()
            # --------------------------------------

        # back to the original model
        basemodel_idx += 1
        if basemodel_idx < len(basemodel_paths):
            print("--> back to the original model: {}".format(ckpt_path))
            model = model_load_ckpt(model, path=ckpt_path)

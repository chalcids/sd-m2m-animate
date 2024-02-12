import os.path
import time
import torch


from datetime import datetime
from tqdm import tqdm
import numpy as np

import cv2
import skimage
from PIL import Image, ImageOps, ImageChops
import modules
import modules.images as sdimages
from modules import shared, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts, state
from modules.ui import plaintext_to_html
import modules.scripts as scripts

from scripts.m2m_animate_util import create_folders, get_mov_all_images, images_to_video, save_images, save_image, save_video_settings
from scripts.m2m_animate_config import m2m_animate_output_dir, m2m_animate_export_frames,m2m_animate_save_mask,m2m_animate_enable_mask
from scripts.raft_utils import RAFT_clear_memory,generate_mask


scripts_m2m_animate = scripts.ScriptRunner()

def save_video(images, fps,path,file_name, extension='.mp4'):
    if not os.path.exists(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir)):
        os.makedirs(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir), exist_ok=True)

    r_f = extension

    print(f'Start generating {r_f} file')

    video = images_to_video(images, fps,os.path.join(path, file_name+"_final"+ r_f, ))
    print(f'The generation is complete, the directory::{video}')

    return video


def process_m2m_animate(p, gen_dict,mov_file, movie_frames, max_frames, enable_hr, hr_scale,hr_upscaler, w, h, occlusion_mask_blur,occlusion_mask_flow_multiplier, occlusion_mask_difo_multiplier,occlusion_mask_difs_multiplier,occlusion_mask_trailing,blend_alpha, args):
    processing.fix_seed(p)
    processing_start_time = datetime.now()
    frames_preprocess, frames_postprocess,frames_mask, main_path, file_name = create_folders(mov_file,processing_start_time)
    images = get_mov_all_images(mov_file, movie_frames)
    if(shared.opts.data.get("m2m_animate_export_frames", m2m_animate_export_frames) == True):
        print("Saving the Preprocessed Frames")
        save_images(images,frames_preprocess)

    if not images:
        print('Failed to parse the video, please check')
        return

    print(f'The video conversion is completed, images:{len(images)}')
    if max_frames == -1 or max_frames > len(images):
        max_frames = len(images)

    max_frames = int(max_frames)

    if(shared.opts.data.get("m2m_animate_enable_mask", m2m_animate_enable_mask) == True):
        state.job_count = max_frames * 2  # * p.n_iter
    else:
        state.job_count = max_frames
    generate_images = []
    prev_frame_alpha_mask = None
    prev_img = None
    prev_gen_img = None
    mask = None
    
    print(f"Seed:{p.seed}")
    seed_info = {
        "seed": p.seed,
        "subseed":p.subseed,
        "subseed_strength":p.subseed_strength,
        "seed_resize_from_h":p.seed_resize_from_h,
        "seed_resize_from_w":p.seed_resize_from_w,
        "seed_enable_extras":p.seed_enable_extras
    }
    for i, image in enumerate(images):
        if i >= max_frames:
            break
        
        if(shared.opts.data.get("m2m_animate_enable_mask", m2m_animate_enable_mask) == True):
            state.job = f"{i + 1} out of {(max_frames * 2)}"
        else:
            state.job = f"{i + 1} out of {max_frames}"
        
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')

        if(shared.opts.data.get("m2m_animate_enable_mask", m2m_animate_enable_mask) == True):
            if(i > 0):
                init_img, mask,prev_frame_alpha_mask,alpha_mask,warped_styled_frame = generate_mask(i,img,prev_img,prev_gen_img,prev_frame_alpha_mask,occlusion_mask_blur,occlusion_mask_flow_multiplier,occlusion_mask_difo_multiplier,occlusion_mask_difs_multiplier,occlusion_mask_trailing,blend_alpha,frames_mask)
                pmask = create_processor(gen_dict,mask,seed_info)
                pmask.init_images = [init_img] * pmask.batch_size
                procmask = scripts_m2m_animate.run(pmask, *args)
                if procmask is None:
                    print(f'current progress: {i + 1}/{(max_frames * 2)}')
                    processed_mask = process_images(pmask)
                    processed_frame = np.array(processed_mask.images[0])[...,:3]
                    # normalizing the colors
                    processed_frame = sdimages.resize_image(0, Image.fromarray(processed_frame), warped_styled_frame.shape[1], warped_styled_frame.shape[0], upscaler_name=hr_upscaler)
                    processed_frame = skimage.exposure.match_histograms(np.array(processed_frame),np.array(img) , channel_axis=None)
                    processed_frame = processed_frame.astype(float) * alpha_mask + warped_styled_frame.astype(float) * (1 - alpha_mask)
                    processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
                    init_img = Image.fromarray(processed_frame)
                    if(shared.opts.data.get("m2m_animate_save_mask", m2m_animate_save_mask) == True):
                        save_image(init_img,i,frames_mask,f"_mask_{i}")
            else:
                init_img = img
        else:
            init_img = img
        
        if(shared.opts.data.get("m2m_animate_enable_mask", m2m_animate_enable_mask) == True):
            state.job = f"{i + 2} out of {(max_frames * 2)}"
        print("\nGenerating Image")
        p.init_images = [init_img] * p.batch_size
        proc = scripts_m2m_animate.run(p, *args)
        if proc is None:
            if(shared.opts.data.get("m2m_animate_enable_mask", m2m_animate_enable_mask) == True):
                print(f'current progress: {i + 2}/{(max_frames * 2)}')
            else:
                print(f'current progress: {i + 1}/{max_frames}')
            processed = process_images(p)
            gen_image = processed.images[0]
            prev_img = init_img.copy()
            prev_gen_img = gen_image.copy()
            if(i > 0):
                if(enable_hr and not state.interrupted):
                    print("\nScaling Image")
                    new_width = int(w * hr_scale)
                    new_height = int(h * hr_scale)
                    gen_image = sdimages.resize_image(0, gen_image, new_width, new_height, upscaler_name=hr_upscaler)
                save_image(gen_image,i,frames_postprocess)
                generate_images.append(gen_image)

    video = save_video(generate_images, movie_frames,main_path,file_name)
    settings_dict = {
        "prompt":gen_dict["prompt"],
        "negative_prompt":gen_dict["negative_prompt"],
        "styles":gen_dict["prompt_styles"],
        "sampler_name":gen_dict["sampler_name"],
        "seed":seed_info["seed"],
        "subseed":seed_info["subseed"],
        "subseed_strength":seed_info["subseed_strength"],
        "seed_resize_from_h":seed_info["seed_resize_from_h"],
        "seed_resize_from_w":seed_info["seed_resize_from_w"],
        "seed_enable_extras":seed_info["seed_enable_extras"],
        "steps":gen_dict["steps"],
        "cfg_scale":gen_dict["cfg_scale"],
        "width":gen_dict["width"],
        "height":gen_dict["height"],
        "resize_mode":gen_dict["resize_mode"],
        "denoising_strength":gen_dict["denoising_strength"],
        "image_cfg_scale":gen_dict["image_cfg_scale"],
        "noise_multiplier":gen_dict["noise_multiplier"],
        "mask_blur":4,
        "inpainting_fill":1,
        "inpaint_full_res":False,
        "inpaint_full_res_padding":32,
        "inpainting_mask_invert":0,
    }
    if(enable_hr):      
        new_width = int(w * hr_scale)
        new_height = int(h * hr_scale)
        settings_dict["new_width"] = new_width
        settings_dict["new_height"] = new_height
        settings_dict["hr_upscaler"] = hr_upscaler
    save_video_settings(settings_dict,main_path)
    RAFT_clear_memory()
    torch.cuda.empty_cache()
    # Only use following if not working with multiple processes sharing GPU mem
    # Ensures that all unneeded IPC handles are released and that GPU memory is being used efficiently
    torch.cuda.ipc_collect()
    print("GPU cache has been cleared.")
    return video

def animate(id_task: str,
            prompt,
            negative_prompt,
            prompt_styles,
            mov_file,
            steps,
            sampler_name,
            cfg_scale,
            image_cfg_scale,
            denoising_strength,
            height,
            width,
            resize_mode,
            override_settings_texts,

            noise_multiplier,
            movie_frames,
            max_frames,
            enable_hr,
            hr_scale,
            hr_upscaler,
            occlusion_mask_blur,
            occlusion_mask_flow_multiplier,
            occlusion_mask_difo_multiplier,
            occlusion_mask_difs_multiplier,
            occlusion_mask_trailing,
            blend_alpha,
            *args):
    if not mov_file:
        raise Exception('Error！ Please add a video file!')

    override_settings = create_override_settings_dict(override_settings_texts)
    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    gen_dict = {
        "prompt":prompt,
        "negative_prompt":negative_prompt,
        "prompt_styles":prompt_styles,
        "sampler_name":sampler_name,
        "steps":steps,
        "cfg_scale":cfg_scale,
        "width":width,
        "height":height,
        "resize_mode":resize_mode,
        "denoising_strength":denoising_strength,
        "image_cfg_scale":image_cfg_scale,
        "noise_multiplier":noise_multiplier,
        "mask_blur":4,
        "inpainting_fill":1,
        "inpaint_full_res":False,
        "inpaint_full_res_padding":32,
        "inpainting_mask_invert":0,
        "override_settings":override_settings,
        "args":args
    }

    p = create_processor(gen_dict,None)

    print(f'\nStart parsing the number of mov frames')
    generate_video = process_m2m_animate(p,gen_dict, mov_file, movie_frames, max_frames, enable_hr, hr_scale,hr_upscaler, width, height, occlusion_mask_blur,occlusion_mask_flow_multiplier,occlusion_mask_difo_multiplier,occlusion_mask_difs_multiplier,occlusion_mask_trailing,blend_alpha, args)
    processed = Processed(p, [], p.seed, "")
    
    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return generate_video, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(
        processed.comments, classname="comments")

def create_processor(gen_dict,mask,seed_info=None):
    if(seed_info):
        p = StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            do_not_save_samples=True,
            do_not_save_grid=True,
            prompt=gen_dict["prompt"],
            negative_prompt=gen_dict["negative_prompt"],
            styles=gen_dict["prompt_styles"],
            sampler_name=gen_dict["sampler_name"],
            batch_size=1,
            n_iter=1,
            steps=gen_dict["steps"],
            cfg_scale=gen_dict["cfg_scale"],
            width=gen_dict["width"],
            height=gen_dict["height"],
            init_images=[None],
            mask=mask,
            seed=seed_info["seed"],
            subseed=seed_info["subseed"],
            subseed_strength=seed_info["subseed_strength"],
            seed_resize_from_h=seed_info["seed_resize_from_h"],
            seed_resize_from_w=seed_info["seed_resize_from_w"],
            seed_enable_extras=seed_info["seed_enable_extras"],
            mask_blur=gen_dict["mask_blur"],
            inpainting_fill=gen_dict["inpainting_fill"],
            resize_mode=gen_dict["resize_mode"],
            denoising_strength=gen_dict["denoising_strength"],
            image_cfg_scale=gen_dict["image_cfg_scale"],
            inpaint_full_res=gen_dict["inpaint_full_res"],
            inpaint_full_res_padding=gen_dict["inpaint_full_res_padding"],
            inpainting_mask_invert=gen_dict["inpainting_mask_invert"],
            override_settings=gen_dict["override_settings"],
            initial_noise_multiplier=gen_dict["noise_multiplier"]
        )
    else:
        p = StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            do_not_save_samples=True,
            do_not_save_grid=True,
            prompt=gen_dict["prompt"],
            negative_prompt=gen_dict["negative_prompt"],
            styles=gen_dict["prompt_styles"],
            sampler_name=gen_dict["sampler_name"],
            batch_size=1,
            n_iter=1,
            steps=gen_dict["steps"],
            cfg_scale=gen_dict["cfg_scale"],
            width=gen_dict["width"],
            height=gen_dict["height"],
            init_images=[None],
            mask=mask,

            mask_blur=gen_dict["mask_blur"],
            inpainting_fill=gen_dict["inpainting_fill"],
            resize_mode=gen_dict["resize_mode"],
            denoising_strength=gen_dict["denoising_strength"],
            image_cfg_scale=gen_dict["image_cfg_scale"],
            inpaint_full_res=gen_dict["inpaint_full_res"],
            inpaint_full_res_padding=gen_dict["inpaint_full_res_padding"],
            inpainting_mask_invert=gen_dict["inpainting_mask_invert"],
            override_settings=gen_dict["override_settings"],
            initial_noise_multiplier=gen_dict["noise_multiplier"]
        )
    p.scripts = scripts_m2m_animate
    p.script_args = gen_dict["args"]

    p.extra_generation_params["Mask blur"] = gen_dict["mask_blur"]
    return p
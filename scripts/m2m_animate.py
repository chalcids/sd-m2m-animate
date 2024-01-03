import os.path
import time

from datetime import datetime
from tqdm import tqdm

import cv2
from PIL import Image
import modules
import modules.images as sdimages
from modules import shared, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts, state
from modules.ui import plaintext_to_html
import modules.scripts as scripts

from scripts.app_util import create_folders, get_mov_all_images, images_to_video, save_images, save_image
from scripts.app_config import m2m_animate_output_dir, m2m_animate_export_frames


scripts_m2m_animate = scripts.ScriptRunner()

def save_video(images, fps,path,file_name, extension='.mp4'):
    if not os.path.exists(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir)):
        os.makedirs(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir), exist_ok=True)

    r_f = extension

    print(f'Start generating {r_f} file')

    video = images_to_video(images, fps,os.path.join(path, file_name+"_final"+ r_f, ))
    print(f'The generation is complete, the directory::{video}')

    return video


def process_m2m_animate(p, mov_file, movie_frames, max_frames, enable_hr, hr_scale,hr_upscaler, w, h, args):
    processing.fix_seed(p)
    processing_start_time = datetime.now()
    frames_preprocess, frames_postprocess, main_path, file_name = create_folders(mov_file,processing_start_time)
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

    p.do_not_save_grid = True
    state.job_count = max_frames  # * p.n_iter
    generate_images = []
    for i, image in enumerate(images):
        if i >= max_frames:
            break

        state.job = f"{i + 1} out of {max_frames}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')

        p.init_images = [img] * p.batch_size
        proc = scripts_m2m_animate.run(p, *args)
        if proc is None:
            print(f'current progress: {i + 1}/{max_frames}')
            processed = process_images(p)
            gen_image = processed.images[0]
            if(enable_hr):
                print("Scaling Image")
                new_width = w * hr_scale
                new_height = h * hr_scale
                gen_image = sdimages.resize_image(0, gen_image, new_width, new_height, upscaler_name=hr_upscaler)
            save_image(gen_image,i,frames_postprocess)
            generate_images.append(gen_image)

    video = save_video(generate_images, movie_frames,main_path,file_name)

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
            *args):
    if not mov_file:
        raise Exception('Error！ Please add a video file!')

    override_settings = create_override_settings_dict(override_settings_texts)
    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    mask_blur = 4
    inpainting_fill = 1
    inpaint_full_res = False
    inpaint_full_res_padding = 32
    inpainting_mask_invert = 0

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        do_not_save_samples=True,
        do_not_save_grid=True,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        sampler_name=sampler_name,
        batch_size=1,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[None],
        mask=None,

        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        initial_noise_multiplier=noise_multiplier

    )

    p.scripts = scripts_m2m_animate
    p.script_args = args

    p.extra_generation_params["Mask blur"] = mask_blur

    print(f'\nStart parsing the number of mov frames')
    generate_video = process_m2m_animate(p, mov_file, movie_frames, max_frames, enable_hr, hr_scale,hr_upscaler, width, height, args)
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

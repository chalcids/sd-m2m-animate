import os.path
import platform
import cv2
import numpy
import imageio
import PIL.Image
import json 
from tqdm import tqdm

from modules import shared
from modules.shared import state
from modules.paths_internal import extensions_dir
from scripts.m2m_animate_config import m2m_animate_output_dir, m2m_animate_export_frames,m2m_animate_save_mask


def calc_video_w_h(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Can't open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return width, height

def calc_video_frames(video_path):
    fps = get_mov_fps(video_path)
    frames = get_mov_frame_count(video_path)
    return fps, frames


def get_mov_frame_count(file):
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames


def get_mov_fps(file):
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_mov_all_images(file, frames, rgb=False):
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if frames > fps:
        print('Waring: The set number of frames is greater than the number of video frames')
        frames = int(fps)

    skip = fps // frames
    count = 1
    fs = 1
    image_list = []
    while (True):
        flag, frame = cap.read()
        if not flag:
            break
        else:
            if fs % skip == 0:
                if rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_list.append(frame)
                if(count == 1):
                    image_list.append(frame)
                count += 1
        fs += 1
    cap.release()
    return image_list


def images_to_video(images, frames, out_path):
    if platform.system() == 'Windows':
        # Use imageio with the 'libx264' codec on Windows
        return images_to_video_imageio(images, frames, out_path, 'libx264')
    elif platform.system() == 'Darwin':
        # Use cv2 with the 'avc1' codec on Mac
        return images_to_video_cv2(images, frames, out_path, 'avc1')
    else:
        # Use cv2 with the 'mp4v' codec on other operating systems as it's the most widely supported
        return images_to_video_cv2(images, frames, out_path, 'mp4v')


def images_to_video_imageio(images, frames, out_path, codec):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with imageio.v2.get_writer(out_path, format='ffmpeg', mode='I', fps=frames, codec=codec) as writer:
        for img in images:
            writer.append_data(numpy.asarray(img))
    return out_path


def images_to_video_cv2(images, frames, out_path, codec):
    if len(images) <= 0:
        return None
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    if len(images) > 0:
        img = images[0]
        img_width, img_height = img.size
        w = img_width
        h = img_height
    video = cv2.VideoWriter(out_path, fourcc, frames, (w, h))
    for image in images:
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()
    return out_path

def create_folders(video, start_date):
    if not os.path.exists(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir)):
        os.mkdir(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir))
    file_name = os.path.basename(video)
    file_names = os.path.splitext(file_name)
    file_name = file_names[0].replace(" ", "")
    fileName = file_name
    if(len(file_name) > 10):
        file_name = file_name[0:10]
    file_name = file_name + "_" + start_date.strftime("%Y%m%d")
    subfolders= [f for f in os.listdir(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir)) if os.path.isdir(os.path.join(shared.opts.data.get("m2m_animate_output_dir", m2m_animate_output_dir), f))]
    duplicatesFound = 1
    for dirname in list(subfolders):
        #print(dirname)
        if(file_name.lower() in dirname.lower()):
            duplicatesFound += 1
    file_name = f"{file_name}_{duplicatesFound}"
    print(f'Project Folder: {file_name}')
    main_path = f"{shared.opts.data.get('m2m_animate_output_dir', m2m_animate_output_dir)}/{file_name}"
    if not os.path.exists(main_path):
           os.mkdir(main_path)
    if(shared.opts.data.get("m2m_animate_export_frames", m2m_animate_export_frames) == True):
        frames_preprocess = f"{main_path}/frames_export"
        if not os.path.exists(frames_preprocess):
            os.mkdir(frames_preprocess)
    else:
        frames_preprocess = None
    if(shared.opts.data.get("m2m_animate_save_mask", m2m_animate_save_mask) == True):
        frames_mask = f"{main_path}/frames_mask"
        if not os.path.exists(frames_mask):
            os.mkdir(frames_mask)
    else:
        frames_mask = None
    frames_postprocess = f"{main_path}/frames_generated"
    if not os.path.exists(frames_postprocess):
           os.mkdir(frames_postprocess)
    return frames_preprocess, frames_postprocess,frames_mask,main_path, fileName

def save_images(images,path):
    #print(f"Saving Images to: {path}")
    for i, image in tqdm(enumerate(images),f"Currently saving Images",len(images)):
        if state.interrupted:
            break
        img = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        save_image(img,(i+1),path)
    return

def save_image(image,i,path,extra=""):
    image.save(f"{path}/frame{extra}_{i}.png")
    return

def save_video_settings(dict,path):
    # Convert and write JSON object to file
    with open(f"{path}/settings.json", "w") as outfile: 
        json.dump(dict, outfile)

def save_settings(prompt,neg_prompt,height,width,steps,sampler_name,cfg_scale,denoising_strength,noise_multiplier,enable_hr,hr_scale,hr_upscaler,seed,subseed):
    #print(f"{extensions_dir}\sd-m2m-animate\config.json")
    settings_dict = {
        "prompt":prompt,
        "neg_prompt":neg_prompt,
        "width":width,
        "height":height,
        "steps":steps,
        "sampler_name":sampler_name,
        "cfg_scale":cfg_scale,
        "denoising_strength":denoising_strength,
        "noise_multiplier":noise_multiplier,
        "enable_hr":enable_hr,
        "hr_scale":hr_scale,
        "hr_upscaler":hr_upscaler,
        "seed":seed,
        "subseed":subseed
    }
    with open(f"{extensions_dir}\sd-m2m-animate\config.json", "w") as outfile: 
        json.dump(settings_dict, outfile)

def load_settings(prompt,neg_prompt,height,width,steps,sampler_name,cfg_scale,denoising_strength,noise_multiplier,enable_hr,hr_scale,hr_upscaler,seed,subseed):
    #print(f"{extensions_dir}\sd-m2m-animate\config.json")
    with open(f"{extensions_dir}\sd-m2m-animate\config.json") as outfile: 
        settings = json.loads(outfile.read())
        prompt = settings['prompt']
        neg_prompt = settings['neg_prompt']
        width = settings['width']
        height = settings['height']
        steps = settings['steps']
        sampler_name = settings['sampler_name']
        cfg_scale = settings['cfg_scale']
        denoising_strength = settings['denoising_strength']
        noise_multiplier = settings['noise_multiplier']
        enable_hr = settings['enable_hr']
        hr_scale = settings['hr_scale']
        hr_upscaler = settings['hr_upscaler']
        seed = settings['seed']
        subseed = settings['subseed']
    return prompt,neg_prompt,height, width, steps,sampler_name,cfg_scale,denoising_strength,noise_multiplier,enable_hr,hr_scale,hr_upscaler,seed,subseed

def save_seed_settings(seed,subseed):
    with open(f"{extensions_dir}\sd-m2m-animate\config.json") as infile: 
        settings = json.loads(infile.read())
        settings['seed'] = seed
        settings['subseed'] = subseed
    with open(f"{extensions_dir}\sd-m2m-animate\config.json", "w") as outfile: 
        json.dump(settings, outfile)
    return

def load_prev_seed(type,element):
    with open(f"{extensions_dir}\sd-m2m-animate\config.json") as infile: 
        settings = json.loads(infile.read())
        element = settings[type]
    return element
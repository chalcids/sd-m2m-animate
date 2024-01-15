import importlib
import os
import platform
import shutil
import subprocess as sp
import sys

import gradio as gr

import modules
import modules.scripts as scripts
from modules import (
    script_callbacks,
    shared,
    call_queue,
    processing,
    sd_samplers,
    ui_prompt_styles,
    sd_models,
)
from modules.call_queue import wrap_gradio_gpu_call
from modules.images import image_data
from modules.shared import opts
from modules.ui import (
    ordered_ui_categories,
    create_sampler_and_steps_selection,
    switch_values_symbol,
    create_override_settings_dropdown,
    detect_image_size_symbol,
    plaintext_to_html,
    paste_symbol,
    clear_prompt_symbol,
    restore_progress_symbol,
)
from modules.ui_common import (
    folder_symbol,
    update_generation_info,
    create_refresh_button,
)
from modules.ui_components import (
    ResizeHandleRow,
    FormRow,
    ToolButton,
    FormHTML,
    FormGroup,
    InputAccordion,
)
from scripts import app_hook as patches
from scripts import app_util
from scripts import m2m_animate
from scripts.m2m_animate import scripts_m2m_animate
from scripts.app_config import m2m_animate_output_dir, m2m_animate_export_frames, m2m_animate_save_mask,m2m_animate_enable_mask

id_part = "m2m_animate"


def save_video(video):
    path = "logs/movies"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    index = len([path for path in os.listdir(path) if path.endswith(".mp4")]) + 1
    video_path = os.path.join(path, str(index).zfill(5) + ".mp4")
    shutil.copyfile(video, video_path)
    filename = os.path.relpath(video_path, path)
    return gr.File.update(value=video_path, visible=True), plaintext_to_html(
        f"Saved: {filename}"
    )


class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    def __init__(self, is_img2img, id_part=None):
        if not id_part:
            id_part = "img2img" if is_img2img else "txt2img"
        self.id_part = id_part

        with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
            with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.prompt = gr.Textbox(
                                label="Prompt",
                                elem_id=f"{id_part}_prompt",
                                show_label=False,
                                lines=3,
                                placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)",
                                elem_classes=["prompt"],
                            )
                            self.prompt_img = gr.File(
                                label="",
                                elem_id=f"{id_part}_prompt_image",
                                file_count="single",
                                type="binary",
                                visible=False,
                            )

                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.negative_prompt = gr.Textbox(
                                label="Negative prompt",
                                elem_id=f"{id_part}_neg_prompt",
                                show_label=False,
                                lines=3,
                                placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)",
                                elem_classes=["prompt"],
                            )

            self.button_interrogate = None
            self.button_deepbooru = None
            if is_img2img:
                with gr.Column(scale=1, elem_classes="interrogate-col"):
                    self.button_interrogate = gr.Button(
                        "Interrogate\nCLIP", elem_id="interrogate"
                    )
                    self.button_deepbooru = gr.Button(
                        "Interrogate\nDeepBooru", elem_id="deepbooru"
                    )

            with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
                with gr.Row(
                    elem_id=f"{id_part}_generate_box", elem_classes="generate-box"
                ):
                    self.interrupt = gr.Button(
                        "Interrupt",
                        elem_id=f"{id_part}_interrupt",
                        elem_classes="generate-box-interrupt",
                    )
                    self.skip = gr.Button(
                        "Skip",
                        elem_id=f"{id_part}_skip",
                        elem_classes="generate-box-skip",
                    )
                    self.submit = gr.Button(
                        "Generate", elem_id=f"{id_part}_generate", variant="primary"
                    )

                    self.skip.click(
                        fn=lambda: shared.state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    self.interrupt.click(
                        fn=lambda: shared.state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )

                with gr.Row(elem_id=f"{id_part}_tools"):
                    self.paste = ToolButton(value=paste_symbol, elem_id="paste")

                    self.clear_prompt_button = ToolButton(
                        value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt"
                    )
                    self.restore_progress_button = ToolButton(
                        value=restore_progress_symbol,
                        elem_id=f"{id_part}_restore_progress",
                        visible=False,
                    )

                    self.token_counter = gr.HTML(
                        value="<span>0/75</span>",
                        elem_id=f"{id_part}_token_counter",
                        elem_classes=["token-counter"],
                    )
                    self.token_button = gr.Button(
                        visible=False, elem_id=f"{id_part}_token_button"
                    )
                    self.negative_token_counter = gr.HTML(
                        value="<span>0/75</span>",
                        elem_id=f"{id_part}_negative_token_counter",
                        elem_classes=["token-counter"],
                    )
                    self.negative_token_button = gr.Button(
                        visible=False, elem_id=f"{id_part}_negative_token_button"
                    )

                    self.clear_prompt_button.click(
                        fn=lambda *x: x,
                        _js="confirm_clear_prompt",
                        inputs=[self.prompt, self.negative_prompt],
                        outputs=[self.prompt, self.negative_prompt],
                    )

                self.ui_styles = ui_prompt_styles.UiPromptStyles(
                    id_part, self.prompt, self.negative_prompt
                )

        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )


def create_output_panel(tabname, outdir):
    def open_folder(f):
        if not os.path.exists(f):
            print(
                f'Folder "{f}" does not exist. After you create an image, the folder will be created.'
            )
            return
        elif not os.path.isdir(f):
            print(
                f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""",
                file=sys.stderr,
            )
            return

        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                sp.Popen(["wsl-open", path])
            else:
                sp.Popen(["xdg-open", path])

    with gr.Column(variant="panel", elem_id=f"{tabname}_results"):
        with gr.Group(elem_id=f"{tabname}_gallery_container"):
            result_gallery = gr.Gallery(
                label="Output",
                show_label=False,
                elem_id=f"{tabname}_gallery",
                columns=4,
                preview=True,
                height=shared.opts.gallery_height or None,
            )
            result_video = gr.PlayableVideo(
                label="Output Video", show_label=False, elem_id=f"{tabname}_video"
            )

        generation_info = None
        with gr.Column():
            with gr.Row(
                elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"
            ):
                open_folder_button = ToolButton(
                    folder_symbol,
                    elem_id=f"{tabname}_open_folder",
                    visible=not shared.cmd_opts.hide_ui_dir_config,
                    tooltip="Open images output directory.",
                )

                if tabname != "extras":
                    save = ToolButton(
                        "💾",
                        elem_id=f"save_{tabname}",
                        tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).",
                    )

            open_folder_button.click(
                fn=lambda: open_folder(shared.opts.outdir_samples or outdir),
                inputs=[],
                outputs=[],
            )

            download_files = gr.File(
                None,
                file_count="multiple",
                interactive=False,
                show_label=False,
                visible=False,
                elem_id=f"download_files_{tabname}",
            )

            with gr.Group():
                html_info = gr.HTML(
                    elem_id=f"html_info_{tabname}", elem_classes="infotext"
                )
                html_log = gr.HTML(
                    elem_id=f"html_log_{tabname}", elem_classes="html-log"
                )

                generation_info = gr.Textbox(
                    visible=False, elem_id=f"generation_info_{tabname}"
                )
                if tabname == "txt2img" or tabname == "img2img" or tabname == "m2m_animate":
                    generation_info_button = gr.Button(
                        visible=False, elem_id=f"{tabname}_generation_info_button"
                    )
                    generation_info_button.click(
                        fn=update_generation_info,
                        _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                        inputs=[generation_info, html_info, html_info],
                        outputs=[html_info, html_info],
                        show_progress=False,
                    )

                save.click(
                    fn=call_queue.wrap_gradio_call(save_video),
                    inputs=[result_video],
                    outputs=[
                        download_files,
                        html_log,
                    ],
                    show_progress=False,
                )

            return result_gallery, result_video, generation_info, html_info, html_log

def on_ui_tabs():
    scripts_m2m_animate.initialize_scripts(is_img2img=True)

    # with gr.Blocks(analytics_enabled=False) as m2m_animate_interface:
    with gr.TabItem(
        "M2M Animate", id=f"tab_{id_part}", elem_id=f"tab_{id_part}"
    ) as m2m_animate_interface:
        toprow = Toprow(is_img2img=False, id_part=id_part)
        dummy_component = gr.Label(visible=False)
        with gr.Tab(
            "Generation", id=f"{id_part}_generation"
        ) as m2m_animate_generation_tab, ResizeHandleRow(equal_height=False):
            with gr.Column(variant="compact", elem_id="m2m_animate_settings"):
                with gr.Tabs(elem_id=f"mode_{id_part}"):
                    init_mov = gr.Video(
                        label="Video for M2M Animate",
                        elem_id=f"{id_part}_mov",
                        show_label=True,
                        source="upload",
                    )

                with FormRow():
                    resize_mode = gr.Radio(
                        label="Resize mode",
                        elem_id=f"{id_part}_resize_mode",
                        choices=[
                            "Just resize",
                            "Crop and resize",
                            "Resize and fill",
                        ],
                        type="index",
                        value="Just resize",
                    )
                scripts_m2m_animate.prepare_ui()

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_name = create_sampler_and_steps_selection(
                            sd_samplers.visible_sampler_names(), id_part
                        )
                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id=f"{id_part}_column_size", scale=4):
                                with gr.Tabs():
                                    with gr.Tab(
                                        label="Resize to",
                                        elem_id=f"{id_part}_tab_resize_to",
                                    ) as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(
                                                elem_id=f"{id_part}_column_size",
                                                scale=4,
                                            ):
                                                width = gr.Slider(
                                                    minimum=64,
                                                    maximum=2048,
                                                    step=8,
                                                    label="Width",
                                                    value=512,
                                                    elem_id=f"{id_part}_width",
                                                )
                                                height = gr.Slider(
                                                    minimum=64,
                                                    maximum=2048,
                                                    step=8,
                                                    label="Height",
                                                    value=512,
                                                    elem_id=f"{id_part}_height",
                                                )
                                            with gr.Column(
                                                elem_id=f"{id_part}_dimensions_row",
                                                scale=1,
                                                elem_classes="dimensions-tools",
                                            ):
                                                res_switch_btn = ToolButton(
                                                    value=switch_values_symbol,
                                                    elem_id=f"{id_part}_res_switch_btn",
                                                )
                                                detect_image_size_btn = ToolButton(
                                                    value=detect_image_size_symbol,
                                                    elem_id=f"{id_part}_detect_image_size_btn",
                                                )
                                    with gr.Tab(label="Upscale Final Generation", elem_id=f"{id_part}_tab_upscale_gen_final",) as tab_upscale_gen_final:
                                        with FormRow(elem_id="m2m_animate_hires_fix_row0", variant="compact"):
                                            enable_hr = gr.Checkbox(label="Enable Upscaling Video")
                                            hr_final_resolution = FormHTML(value="", elem_id="m2m_animate_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)

                                        with FormRow(elem_id="m2m_animate_hires_fix_row1", variant="compact"):
                                            hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="m2m_animate_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=1.0, elem_id="m2m_animate_hr_scale")    
                    elif category == "denoising":
                        denoising_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            label="Denoising strength",
                            value=0.75,
                            elem_id=f"{id_part}_denoising_strength",
                        )

                        noise_multiplier = gr.Slider(
                            minimum=0,
                            maximum=1.5,
                            step=0.01,
                            label="Noise multiplier",
                            elem_id=f"{id_part}_noise_multiplier",
                            value=1,
                        )
                        with gr.Row(elem_id=f"{id_part}_frames_setting"):
                            movie_frames = gr.Slider(
                                minimum=10,
                                maximum=60,
                                step=1,
                                label="Movie FPS",
                                elem_id=f"{id_part}_movie_frames",
                                value=30,
                            )
                            max_frames = gr.Number(
                                label="Max FPS",
                                value=-1,
                                elem_id=f"{id_part}_max_frames",
                            )
                        with gr.Accordion("Extra settings",open=False):
                            gr.HTML('# Occlusion mask params:')
                            with gr.Row():
                                with gr.Column(scale=1, variant='compact'):
                                    occlusion_mask_blur = gr.Slider(label='Occlusion blur strength', minimum=0, maximum=10, step=0.1, value=3, interactive=True) 
                                    blend_alpha = gr.Slider(label='Warped prev frame vs Current frame blend alpha', minimum=0, maximum=1, step=0.1, value=1, interactive=True) 
                                    occlusion_mask_trailing = gr.Checkbox(label="Occlusion trailing", info="Reduce ghosting but adds more flickering to the video", value=True, interactive=True)
                                with gr.Column(scale=1, variant='compact'):
                                    occlusion_mask_flow_multiplier = gr.Slider(label='Occlusion flow multiplier', minimum=0, maximum=10, step=0.1, value=5, interactive=True) 
                                    occlusion_mask_difo_multiplier = gr.Slider(label='Occlusion diff origin multiplier', minimum=0, maximum=10, step=0.1, value=2, interactive=True)
                                    occlusion_mask_difs_multiplier = gr.Slider(label='Occlusion diff styled multiplier', minimum=0, maximum=10, step=0.1, value=0, interactive=True)
                    elif category == "cfg":
                        with gr.Row():
                            cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=30.0,
                                step=0.5,
                                label="CFG Scale",
                                value=7.0,
                                elem_id=f"{id_part}_cfg_scale",
                            )
                            image_cfg_scale = gr.Slider(
                                minimum=0,
                                maximum=3.0,
                                step=0.05,
                                label="Image CFG Scale",
                                value=1.5,
                                elem_id=f"{id_part}_image_cfg_scale",
                                visible=False,
                            )

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            pass

                    elif category == "accordions":
                        with gr.Row(elem_id=f"{id_part}_accordions", elem_classes="accordions"):
                            scripts_m2m_animate.setup_ui_for_section(category)

                    elif category == "override_settings":
                        with FormRow(elem_id=f"{id_part}_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown(
                                "scripts_m2m_animate", row
                            )

                    elif category == "scripts":
                        with FormGroup(elem_id=f"{id_part}_script_container"):
                            custom_inputs = scripts_m2m_animate.setup_ui()

                    if category not in {"accordions"}:
                        scripts_m2m_animate.setup_ui_for_section(category)

            (
                m2m_animate_gallery,
                result_video,
                generation_info,
                html_info,
                html_log,
            ) = create_output_panel(id_part, opts.m2m_animate_output_dir)

            res_switch_btn.click(
                fn=None,
                _js="function(){switchWidthHeight('m2m_animate')}",
                inputs=None,
                outputs=None,
                show_progress=False,
            )

            # calc video size
            detect_image_size_btn.click(
                fn=calc_video_w_h,
                inputs=[init_mov, width, height],
                outputs=[width, height],
            )

            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale]

            for component in hr_resolution_preview_inputs:
                event = component.release if isinstance(component, gr.Slider) else component.change

                event(
                    fn=calc_resolution_hires,
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=True,
                )

            m2m_animate_args = dict(
                fn=wrap_gradio_gpu_call(m2m_animate.animate, extra_outputs=[None, "", ""]),
                _js="submit_m2m_animate",
                inputs=[
                    dummy_component,
                    toprow.prompt,
                    toprow.negative_prompt,
                    toprow.ui_styles.dropdown,
                    init_mov,
                    steps,
                    sampler_name,
                    cfg_scale,
                    image_cfg_scale,
                    denoising_strength,
                    height,
                    width,
                    resize_mode,
                    override_settings,
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
                    blend_alpha
                ]
                + custom_inputs,
                outputs=[
                    result_video,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            toprow.submit.click(**m2m_animate_args)

    return [(m2m_animate_interface, "M2M Animate", f"{id_part}_tabs")]


def calc_video_w_h(video, width, height):
    if not video:
        return width, height

    return app_util.calc_video_w_h(video)


def on_ui_settings():
    section = ("scripts_m2m_animate", "M2M Animate")
    shared.opts.add_option(
        "m2m_animate_output_dir",
        shared.OptionInfo(
            m2m_animate_output_dir, "M2M Animate output path for projects", section=section
        ),
    )
    shared.opts.add_option(
        "m2m_animate_export_frames",
        shared.OptionInfo(
            m2m_animate_export_frames, "Save orginal frames of video in a folder",gr.Checkbox,{"interactive": True}, section=section
        ),
    )
    shared.opts.add_option(
        "m2m_animate_enable_mask",
        shared.OptionInfo(
            m2m_animate_enable_mask, "Use Masking to better enhance the video generation",gr.Checkbox,{"interactive": True}, section=section
        ),
    )
    
    shared.opts.add_option(
        "m2m_animate_save_mask",
        shared.OptionInfo(
            m2m_animate_save_mask, "Save generated masks in a folder",gr.Checkbox,{"interactive": True}, section=section
        ),
    )
    


img2img_toprow: gr.Row = None

def calc_resolution_hires(enable_hr, width, height, hr_scale ):
    if not enable_hr:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=0, hr_resize_y=0)
    p.calculate_target_resolution()
    return f"<span style='float: right;'>from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span></span>"


def block_context_init(self, *args, **kwargs):
    origin_block_context_init(self, *args, **kwargs)

    if self.elem_id == "tab_img2img":
        self.parent.__enter__()
        on_ui_tabs()
        self.parent.__exit__()


def on_app_reload():
    global origin_block_context_init
    if origin_block_context_init:
        patches.undo(__name__, obj=gr.blocks.BlockContext, field="__init__")
        origin_block_context_init = None


origin_block_context_init = patches.patch(
    __name__,
    obj=gr.blocks.BlockContext,
    field="__init__",
    replacement=block_context_init,
)
script_callbacks.on_before_reload(on_app_reload)
script_callbacks.on_ui_settings(on_ui_settings)

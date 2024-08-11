# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
import os
import shlex
import time
from dataclasses import dataclass
from typing import Optional

import gradio as gr
import simple_parsing
import yaml
from einops import rearrange, repeat

import numpy as np
import torch
from torchvision.utils import make_grid

from ml_mdm import helpers, reader
from ml_mdm.config import get_arguments, get_model, get_pipeline
from ml_mdm.language_models import factory

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Note that it is called add_arguments, not add_argument.
logging.basicConfig(
    level=getattr(logging, "INFO", None),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def dividable(n):
    for i in range(int(np.sqrt(n)), 0, -1):
        if n % i == 0:
            break
    return i, n // i


def generate_lm_outputs(device, sample, tokenizer, language_model, args):
    with torch.no_grad():
        lm_outputs, lm_mask = language_model(sample, tokenizer)
        sample["lm_outputs"] = lm_outputs
        sample["lm_mask"] = lm_mask
    return sample


def setup_models(args, device):
    input_channels = 3

    # load the language model
    tokenizer, language_model = factory.create_lm(args, device=device)
    language_model_dim = language_model.embed_dim
    args.unet_config.conditioning_feature_dim = language_model_dim
    denoising_model = get_model(args.model)(
        input_channels, input_channels, args.unet_config
    ).to(device)
    diffusion_model = get_pipeline(args.model)(
        denoising_model, args.diffusion_config
    ).to(device)
    # denoising_model.print_size(args.sample_image_size)
    return tokenizer, language_model, diffusion_model


def plot_logsnr(logsnrs, total_steps):
    import matplotlib.pyplot as plt

    x = 1 - np.arange(len(logsnrs)) / (total_steps - 1)
    plt.plot(x, np.asarray(logsnrs))
    plt.xlabel("timesteps")
    plt.ylabel("LogSNR")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(-20, 10)
    plt.gca().invert_xaxis()

    # Convert the plot to a numpy array
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer._renderer)
    plt.close()
    return image


@dataclass
class GLOBAL_DATA:
    reader_config: Optional[reader.ReaderConfig] = None
    tokenizer = None
    args = None
    language_model = None
    diffusion_model = None
    override_args = ""
    ckpt_name = ""
    config_file = ""


global_config = GLOBAL_DATA()


def stop_run():
    return (
        gr.update(value="Run", variant="primary", visible=True),
        gr.update(visible=False),
    )


def get_model_type(config_file):
    with open(config_file, "r") as f:
        d = yaml.safe_load(f)
        return d.get("model", d.get("vision_model", "unet"))


def generate(
    config_file="cc12m_64x64.yaml",
    ckpt_name="vis_model_64x64.pth",
    prompt="a chair",
    input_template="",
    negative_prompt="",
    negative_template="",
    batch_size=20,
    guidance_scale=7.5,
    threshold_function="clip",
    num_inference_steps=250,
    eta=0,
    save_diffusion_path=False,
    show_diffusion_path=False,
    show_xt=False,
    reader_config="",
    seed=10,
    comment="",
    override_args="",
    output_inner=False,
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if len(input_template) > 0:
        prompt = input_template.format(prompt=prompt)
    if len(negative_template) > 0:
        negative_prompt = negative_prompt + negative_template
    print(f"Postive: {prompt} / Negative: {negative_prompt}")

    if not os.path.exists(ckpt_name):
        logging.info(f"Did not generate because {ckpt_name} does not exist")
        return None, None, f"{ckpt_name} does not exist", None, None

    if (
        global_config.config_file != config_file
        or global_config.ckpt_name != ckpt_name
        or global_config.override_args != override_args
    ):
        # Identify model type
        model_type = get_model_type(f"configs/models/{config_file}")
        # reload the arguments
        args = get_arguments(
            shlex.split(override_args + f" --model {model_type}"),
            mode="demo",
            additional_config_paths=[f"configs/models/{config_file}"],
        )
        helpers.print_args(args)

        # setup model when the parent task changed.
        tokenizer, language_model, diffusion_model = setup_models(args, device)
        vision_model_file = ckpt_name
        try:
            other_items = diffusion_model.model.load(vision_model_file)
        except Exception as e:
            logging.error(f"failed to load {vision_model_file}", exc_info=e)
            return None, None, "Loading Model Error", None, None

        # setup global configs
        global_config.batch_num = -1  # reset batch num
        global_config.args = args
        global_config.override_args = override_args
        global_config.tokenizer = tokenizer
        global_config.language_model = language_model
        global_config.diffusion_model = diffusion_model
        global_config.reader_config = args.reader_config
        global_config.config_file = config_file
        global_config.ckpt_name = ckpt_name

    else:
        args = global_config.args
        tokenizer = global_config.tokenizer
        language_model = global_config.language_model
        diffusion_model = global_config.diffusion_model

    tokenizer = global_config.tokenizer

    sample = {}
    sample["text"] = [negative_prompt, prompt] if guidance_scale != 1 else [prompt]
    sample["tokens"] = np.asarray(
        reader.process_text(sample["text"], tokenizer, args.reader_config)
    )
    sample = generate_lm_outputs(device, sample, tokenizer, language_model, args)
    assert args.sample_image_size != -1

    # set up thresholding
    from samplers import ThresholdType

    diffusion_model.sampler._config.threshold_function = {
        "clip": ThresholdType.CLIP,
        "dynamic (Imagen)": ThresholdType.DYNAMIC,
        "dynamic (DeepFloyd)": ThresholdType.DYNAMIC_IF,
        "none": ThresholdType.NONE,
    }[threshold_function]

    output_comments = f"{comment}\n"

    bsz = batch_size
    with torch.no_grad():
        if bsz > 1:
            sample["lm_outputs"] = repeat(
                sample["lm_outputs"], "b n d -> (b r) n d", r=bsz
            )
            sample["lm_mask"] = repeat(sample["lm_mask"], "b n -> (b r) n", r=bsz)

        num_samples = bsz
        original, outputs, logsnrs = [], [], []
        logging.info(f"Starting to sample from the model")
        start_time = time.time()
        for step, result in enumerate(
            diffusion_model.sample(
                num_samples,
                sample,
                args.sample_image_size,
                device,
                return_sequence=False,
                num_inference_steps=num_inference_steps,
                ddim_eta=eta,
                guidance_scale=guidance_scale,
                resample_steps=True,
                disable_bar=False,
                yield_output=True,
                yield_full=True,
                output_inner=output_inner,
            )
        ):
            x0, x_t, extra = result
            if step < num_inference_steps:
                g = extra[0][0, 0, 0, 0].cpu()
                logsnrs += [torch.log(g / (1 - g))]
            output = x0 if not show_xt else x_t
            output = torch.clamp(output * 0.5 + 0.5, min=0, max=1).cpu()
            original += [
                output if not output_inner else output[..., -args.sample_image_size :]
            ]

            output = (
                make_grid(output, nrow=dividable(bsz)[0]).permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            outputs += [output]

            output_video_path = None
            if step == num_inference_steps and save_diffusion_path:
                import imageio

                writer = imageio.get_writer("temp_output.mp4", fps=32)
                for output in outputs:
                    writer.append_data(output)
                writer.close()
                output_video_path = "temp_output.mp4"
                if any(diffusion_model.model.vision_model.is_temporal):
                    data = rearrange(
                        original[-1],
                        "(a b) c (n h) (m w) -> (n m) (a h) (b w) c",
                        a=dividable(bsz)[0],
                        n=4,
                        m=4,
                    )
                    data = (data.numpy() * 255).astype(np.uint8)
                    writer = imageio.get_writer("temp_output.mp4", fps=4)
                    for d in data:
                        writer.append_data(d)
                    writer.close()

            if show_diffusion_path or (step == num_inference_steps):
                yield output, plot_logsnr(
                    logsnrs, num_inference_steps
                ), output_comments + f"Step ({step} / {num_inference_steps}) Time ({time.time() - start_time:.4}s)", output_video_path, gr.update(
                    value="Run",
                    variant="primary",
                    visible=(step == num_inference_steps),
                ), gr.update(
                    value="Stop", variant="stop", visible=(step != num_inference_steps)
                )


def main(args):
    # get the language model outputs
    example_texts = open("data/prompts_demo.tsv").readlines()

    css = """
        #config-accordion, #logs-accordion {color: black !important;}
        .dark #config-accordion, .dark #logs-accordion {color: white !important;}
        .stop {background: darkred !important;}
        """

    with gr.Blocks(
        title="Demo of Text-to-Image Diffusion",
        theme="EveryPizza/Cartoony-Gradio-Theme",
        css=css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
                    # MLR Text-to-Image Diffusion Model Web Demo

                    ### Usage
                    - Select examples below or manually input model and prompts
                    - Change more advanced settings such as inference steps.
                    """
            gr.Markdown(header)

        with gr.Row(equal_height=False):
            pid = gr.State()
            with gr.Column(scale=2):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        config_file = gr.Dropdown(
                            [
                                "cc12m_64x64.yaml",
                                "cc12m_256x256.yaml",
                                "cc12m_1024x1024.yaml",
                            ],
                            value="cc12m_64x64.yaml",
                            label="Select the config file",
                        )
                    with gr.Column(scale=1):
                        ckpt_name = gr.Dropdown(
                            [
                                "vis_model_64x64.pth",
                                "vis_model_256x256.pth",
                                "vis_model_1024x1024.pth",
                            ],
                            value="vis_model_64x64.pth",
                            label="Load checkpoint",
                        )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        save_diffusion_path = gr.Checkbox(
                            value=True, label="Show diffusion path as a video"
                        )
                        show_diffusion_path = gr.Checkbox(
                            value=False, label="Show diffusion progress"
                        )
                    with gr.Column(scale=1):
                        show_xt = gr.Checkbox(value=False, label="Show predicted x_t")
                        output_inner = gr.Checkbox(
                            value=False,
                            label="Output inner UNet (High-res models Only)",
                        )

            with gr.Column(scale=2):
                prompt_input = gr.Textbox(label="Input prompt")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        guidance_scale = gr.Slider(
                            value=7.5,
                            minimum=0.0,
                            maximum=50,
                            step=0.1,
                            label="Guidance scale",
                        )
                    with gr.Column(scale=1):
                        batch_size = gr.Slider(
                            value=16, minimum=1, maximum=128, step=1, label="Batch size"
                        )

        with gr.Row(equal_height=False):
            comment = gr.Textbox(value="", label="Comments to the model (optional)")

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                output_image = gr.Image(value=None, label="Output image")
            with gr.Column(scale=2):
                output_video = gr.Video(value=None, label="Diffusion Path")

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                with gr.Accordion(
                    "Advanced settings", open=False, elem_id="config-accordion"
                ):
                    input_template = gr.Dropdown(
                        [
                            "",
                            "breathtaking {prompt}. award-winning, professional, highly detailed",
                            "anime artwork {prompt}. anime style, key visual, vibrant, studio anime, highly detailed",
                            "concept art {prompt}. digital artwork, illustrative, painterly, matte painting, highly detailed",
                            "ethereal fantasy concept art of {prompt}. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
                            "cinematic photo {prompt}. 35mm photograph, film, bokeh, professional, 4k, highly detailed",
                            "cinematic film still {prompt}. shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody",
                            "analog film photo {prompt}. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage",
                            "vaporwave synthwave style {prompt}. cyberpunk, neon, vibes, stunningly beautiful, crisp, detailed, sleek, ultramodern, high contrast, cinematic composition",
                            "isometric style {prompt}. vibrant, beautiful, crisp, detailed, ultra detailed, intricate",
                            "low-poly style {prompt}. ambient occlusion, low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
                            "claymation style {prompt}. sculpture, clay art, centered composition, play-doh",
                            "professional 3d model {prompt}. octane render, highly detailed, volumetric, dramatic lighting",
                            "origami style {prompt}. paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
                            "pixel-art {prompt}. low-res, blocky, pixel art style, 16-bit graphics",
                        ],
                        value="",
                        label="Positive Template (by default, not use)",
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            negative_prompt_input = gr.Textbox(
                                value="", label="Negative prompt"
                            )
                        with gr.Column(scale=1):
                            negative_template = gr.Dropdown(
                                [
                                    "",
                                    "anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry",
                                    "photo, deformed, black and white, realism, disfigured, low contrast",
                                    "photo, photorealistic, realism, ugly",
                                    "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
                                    "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
                                    "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
                                    "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
                                    "illustration, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
                                    "deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic",
                                    "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
                                    "ugly, deformed, noisy, low poly, blurry, painting",
                                ],
                                value="",
                                label="Negative Template (by default, not use)",
                            )

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            threshold_function = gr.Dropdown(
                                [
                                    "clip",
                                    "dynamic (Imagen)",
                                    "dynamic (DeepFloyd)",
                                    "none",
                                ],
                                value="dynamic (DeepFloyd)",
                                label="Thresholding",
                            )
                        with gr.Column(scale=1):
                            reader_config = gr.Dropdown(
                                ["configs/datasets/reader_config.yaml"],
                                value="configs/datasets/reader_config.yaml",
                                label="Reader Config",
                            )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            num_inference_steps = gr.Slider(
                                value=50,
                                minimum=1,
                                maximum=2000,
                                step=1,
                                label="# of steps",
                            )
                        with gr.Column(scale=1):
                            eta = gr.Slider(
                                value=0,
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label="DDIM eta",
                            )
                    seed = gr.Slider(
                        value=137,
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        label="Random seed",
                    )
                    override_args = gr.Textbox(
                        value="--reader_config.max_token_length 128 --reader_config.max_caption_length 512",
                        label="Override model arguments (optional)",
                    )

                run_btn = gr.Button(value="Run", variant="primary")
                stop_btn = gr.Button(value="Stop", variant="stop", visible=False)

            with gr.Column(scale=2):
                with gr.Accordion(
                    "Addditional outputs", open=False, elem_id="output-accordion"
                ):
                    with gr.Row(equal_height=True):
                        output_text = gr.Textbox(value=None, label="System output")
                    with gr.Row(equal_height=True):
                        logsnr_fig = gr.Image(value=None, label="Noise schedule")

        run_event = run_btn.click(
            fn=generate,
            inputs=[
                config_file,
                ckpt_name,
                prompt_input,
                input_template,
                negative_prompt_input,
                negative_template,
                batch_size,
                guidance_scale,
                threshold_function,
                num_inference_steps,
                eta,
                save_diffusion_path,
                show_diffusion_path,
                show_xt,
                reader_config,
                seed,
                comment,
                override_args,
                output_inner,
            ],
            outputs=[
                output_image,
                logsnr_fig,
                output_text,
                output_video,
                run_btn,
                stop_btn,
            ],
        )

        stop_btn.click(
            fn=stop_run,
            outputs=[run_btn, stop_btn],
            cancels=[run_event],
            queue=False,
        )
        example0 = gr.Examples(
            [
                ["cc12m_64x64.yaml", "vis_model_64x64.pth", 64, 50, 0],
                ["cc12m_256x256.yaml", "vis_model_256x256.pth", 16, 100, 0],
                ["cc12m_1024x1024.yaml", "vis_model_1024x1024.pth", 4, 250, 1],
            ],
            inputs=[config_file, ckpt_name, batch_size, num_inference_steps, eta],
        )
        example1 = gr.Examples(
            examples=[[t.strip()] for t in example_texts],
            inputs=[prompt_input],
        )

        launch_args = {"server_port": int(args.port), "server_name": "0.0.0.0"}
        demo.queue(default_concurrency_limit=1).launch(**launch_args)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser(description="pre-loading demo")
    parser.add_argument("--port", type=int, default=19231)
    args = parser.parse_known_args()[0]
    main(args)

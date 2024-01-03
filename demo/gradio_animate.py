# adapt from https://github.com/magic-research/magic-animate/blob/main/demo/gradio_animate.py
import argparse
import imageio
import numpy as np
import gradio as gr
from PIL import Image

from demo.animate import AnimateAnyone

animator = AnimateAnyone()

def animate(reference_image, motion_sequence_state, seed, steps, guidance_scale):
    return animator(reference_image, motion_sequence_state, seed, steps, guidance_scale)

with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/guoqincode/AnimateAnyone-unofficial" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >The unofficial AnimateAnyone implementation.</h1>
            <h1 >https://github.com/guoqincode/AnimateAnyone-unofficial</h1>
            <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
        </div>
        </div>
        """)
    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)
    
    with gr.Row():
        reference_image  = gr.Image(label="Reference Image")
        motion_sequence  = gr.Video(format="mp4", label="Motion Sequence")
        
        with gr.Column():
            random_seed         = gr.Textbox(label="Random seed", value=1, info="default: -1")
            sampling_steps      = gr.Textbox(label="Sampling steps", value=25, info="default: 25")
            guidance_scale      = gr.Textbox(label="Guidance scale", value=7.5, info="default: 7.5")
            submit              = gr.Button("Animate")

    def read_video(video):
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']
        return video
    
    def read_image(image, size=512):
        return np.array(Image.fromarray(image).resize((size, size)))
    
    # when user uploads a new video
    motion_sequence.upload(
        read_video,
        motion_sequence,
        motion_sequence
    )
    # when `first_frame` is updated
    reference_image.upload(
        read_image,
        reference_image,
        reference_image
    )
    # when the `submit` button is clicked
    submit.click(
        animate,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
        animation
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            # ["inputs/applications/source_image/91-9wEBqAlS.png", "inputs/applications/driving/dwpose/91-9wEBqAlS.mp4"], 
            # ["inputs/applications/source_image/91C+rpudhdS.png", "inputs/applications/driving/dwpose/91C+rpudhdS.mp4"],
            # ["inputs/applications/source_image/A11UTfKe+tS.png", "inputs/applications/driving/dwpose/A11UTfKe+tS.mp4"],
            # ["inputs/applications/source_image/A17AGFxllwS.png", "inputs/applications/driving/dwpose/A17AGFxllwS.mp4"],
            # ["inputs/applications/source_image/A18fOhsmJWS.png", "inputs/applications/driving/dwpose/A18fOhsmJWS.mp4"],
            # ["inputs/applications/source_image/91HzMhq7eOS.png", "inputs/applications/driving/dwpose/81FyMPk-WIS.mp4"],
            # ["inputs/applications/source_image/A1dLq8J8cjS.png", "inputs/applications/driving/dwpose/91cqgHAeFJS.mp4"],
            # ["inputs/applications/source_image/A1dLq8J8cjS.png", "inputs/applications/driving/dwpose/91hNaP-63aS.mp4"],
            # ["inputs/applications/source_image/81FyMPk-WIS.png", "inputs/applications/driving/dwpose/81FyMPk-WIS.mp4"],

            ["inputs/applications/source_image/91+20mY7UJS.png", "inputs/applications/driving/dwpose/91+20mY7UJS.mp4"],
            ["inputs/applications/source_image/91+bCFG1jOS.png", "inputs/applications/driving/dwpose/91+bCFG1jOS.mp4"],
            ["inputs/applications/source_image/91+bCFG1jOS.png", "inputs/applications/driving/dwpose/91+bCFG1jOS.mp4"],
            ["inputs/applications/source_image/91+bCFG1jOS.png", "inputs/applications/driving/dwpose/00012_dwpose.mp4"],
            ["inputs/applications/source_image/A18fOhsmJWS.png", "inputs/applications/driving/dwpose/00012_dwpose.mp4"],

        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )


demo.launch(share=False)

# python3 -m demo.gradio_animate
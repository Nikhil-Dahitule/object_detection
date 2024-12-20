import warnings
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import cv2
from PIL import Image
import numpy as np
import logging



warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
logging.getLogger("transformers").setLevel(logging.ERROR)
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
processor.patch_size = 14
processor.vision_feature_select_strategy = "mean"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)


def live_stream_captioning():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    caption_window_width = 640
    caption_window_height = 480

    while True:
        user_prompt = input("Enter a prompt for captioning or 'q' to quit: ")
        if user_prompt.lower() == 'q':
            break

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Construct the prompt for the model
        toks = "<image>"  # Only one frame for a single input
        prompt = f"<|im_start|>user{toks}\n{user_prompt}<|im_end|><|im_start|>assistant"

        inputs = processor(
            text=prompt, images=[pil_img], return_tensors="pt"
        ).to(device, model.dtype)

        # Generate caption with explicitly set pad_token_id
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=model.config.eos_token_id
        )

        raw_caption = processor.decode(output[0][2:], skip_special_tokens=True)
        print("Raw caption:", raw_caption)
        caption = raw_caption[len(user_prompt) + 10:]

        def wrap_text(text, width):
            lines = []
            words = text.split()
            line = ""
            for word in words:
                if len(line + word) <= width:
                    line += " " + word if line else word
                else:
                    lines.append(line)
                    line = word
            if line:
                lines.append(line)
            return lines

        caption_lines = wrap_text(caption, width=30)

        caption_block_height = 30 + len(caption_lines) * 30

        caption_block = 255 * np.ones(
            shape=[caption_block_height, caption_window_width, 3], dtype=np.uint8
        )

        # Clear the caption block
        caption_block.fill(255)


        y_offset = 30
        for line in caption_lines:
            cv2.putText(
                caption_block, line, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
            )
            y_offset += 30

        # Resize the video frame
        frame_resized = cv2.resize(frame, (caption_window_width, caption_window_height))

        if caption_block.shape[0] < caption_window_height:
            extra_height = caption_window_height - caption_block.shape[0]
            caption_block = np.vstack(
                [caption_block, 255 * np.ones((extra_height, caption_window_width, 3), dtype=np.uint8)]
            )

        # Resize caption block if needed
        caption_block_resized = cv2.resize(
            caption_block, (caption_window_width, caption_window_height)
        )

        # Concatenate the video frame and the caption block
        combined_frame = cv2.hconcat([frame_resized, caption_block_resized])

        # Display the result
        cv2.imshow('Captured Image with Caption', combined_frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    cap.release()



live_stream_captioning()

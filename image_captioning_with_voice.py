import warnings
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import cv2
from PIL import Image
import logging
import threading
import time
import speech_recognition as sr
from gtts import gTTS
import os

# Suppress warning
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id")
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

captured_frame = None
lock = threading.Lock()
prompt = ""
prompt_ready = threading.Event()
exit_flag = threading.Event()

cv2.namedWindow('Webcam Live Stream', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Webcam Live Stream', cv2.WND_PROP_TOPMOST, 1)

def capture_frame(cap):
    global captured_frame
    while not exit_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        with lock:
            captured_frame = frame


def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something...")
        recognizer.adjust_for_ambient_noise(source)

        try:
            # Listen continuously until timeout or phrase time limit is exceeded
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("Listening timed out. No input detected.")
            return None  # Return None if no input is detected

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")  # Debugging
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


def text_to_audio(text):
    if text:
        speech = gTTS(text=text, lang='en', slow=False)
        speech.save("output.mp3")
        os.system("start output.mp3")
    else:
        print("No text provided to convert to audio.")


def get_prompt():
    global prompt
    prompt_ready.clear()
    prompt = voice_to_text()  # Get prompt via voice
    if prompt and prompt.lower() != 'q':
        prompt_ready.set()
    if prompt and prompt.lower() == 'q':
        exit_flag.set()  # Set the exit flag to stop all threads


def generate_caption():
    global captured_frame, prompt
    while not exit_flag.is_set():
        prompt_ready.wait()  # Wait until the prompt is ready
        if prompt.lower() == 'q':
            break

        with lock:
            if captured_frame is None:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))

        toks = "<image>"
        full_prompt = f"<|im_start|>user{toks}\n{prompt}<|im_end|><|im_start|>assistant"

        print(f"Processing caption for prompt: {prompt}")  # Debugging

        inputs = processor(
            text=full_prompt, images=[pil_img], return_tensors="pt"
        ).to(device, model.dtype)

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=model.config.eos_token_id
        )

        raw_caption = processor.decode(output[0][2:], skip_special_tokens=True)
        caption = raw_caption[len(prompt) + 10:]

        print(f"Generated Caption: {caption}\n")  # Display the generated caption
        text_to_audio(caption)  # Speak the generated caption aloud

        prompt_ready.clear()  # Clear prompt_ready for the next prompt
        print("\nPress 'Enter' to speak the next prompt or say 'q' to quit.")  # Prompt for the next input



def live_stream_captioning():
    global captured_frame, prompt
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Start capture, key press detection, prompt, and caption generation threads
    threading.Thread(target=capture_frame, args=(cap,), daemon=True).start()
    threading.Thread(target=generate_caption, daemon=True).start()

    while not exit_flag.is_set():
        with lock:
            if captured_frame is not None:
                cv2.imshow('Webcam Live Stream', captured_frame)

        # Check for Enter key press to start voice input
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key is pressed
            print("Enter key pressed. Please speak your prompt...")
            get_prompt()

        if key == ord('q'):
            exit_flag.set()
            break

    cap.release()
    cv2.destroyAllWindows()


live_stream_captioning()

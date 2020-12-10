from source.image import load_image, save_image, get_numpy_transform
import torch
from ..models.StylizationModel import StylizationModel
from PIL import Image
import torchvision.transforms as tf
import numpy as np
import cv2
import argparse
from moviepy.editor import VideoFileClip, ImageSequenceClip
import os
import shutil
import time

MODEL_PATH = "results/mosaic/final_model.pth"
# MODEL_PATH = "results/nude/final_model.pth"
NORMALIZE = True
INPUT_PATH = "tmp_james.jpg"
OUTPUT_PATH = "tmp_style_james.jpg"
VIDEO_IN = "test/kiki.flv"
VIDEO_OUT = "result/kiki.mp4"
TMP_DIR = "tmp"
IMAGE_WIDTH = 800  # 767 // 2
IMAGE_HEIGHT = 600  # 1025 // 2

resize = tf.Compose(
    [
        tf.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        tf.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
    ]
)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser(description="Neural Style Transfer")
    arg_parser.add_argument(
        "-v",
        "--video",
        default=False,
        action="store_true",
        help="toggle for style transfer on video stream",
    )
    arg_parser.add_argument(
        "-w",
        "--webcam",
        default=False,
        action="store_true",
        help="toggle for style transfer on webcam stream",
    )
    args = vars(arg_parser.parse_args())

    # Load the pre-trained model.
    model = StylizationModel(NORMALIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model = model.to(device)

    if args["video"]:
        video_clip = VideoFileClip(VIDEO_IN, audio=False)
        audio_clip = AudioFileClip(VIDEO_IN)
        current_directory = os.getcwd()
        tmp_directory = os.path.join(current_directory, TMP_DIR)
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory)

        for i, frame in enumerate(video_clip.iter_frames()):
            frame = get_numpy_transform(frame).unsqueeze(0)
            pred = model(frame.to(device)).cpu().detach().numpy()[0]
            save_image(tmp_directory + "/" + str(i).zfill(5) + ".png", pred)
        
        video = ImageSequenceClip(
            sequence=tmp_directory + "/", fps=video_clip.fps
        )
        video = video.set_audio(audio_clip)
        video.write_videofile(VIDEO_OUT, audio=True)
        shutil.rmtree(tmp_directory + "/")

    elif args["webcam"]:
        # webcam mode, process frames in webcam stream
        cv2.startWindowThread()
        cv2.namedWindow("frame")
        while True:
            # Read an image.
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = load_image(None, Image.fromarray(image))
            image = resize(image.to(device))

            # Evaluate the model.
            t = time.time()
            output_test = model(image).cpu().detach().numpy()
            print(f"Evaluating model took {time.time() - t} seconds.")

            # Send the result to the display.
            output_test = np.transpose(output_test[0, :, :, :], (1, 2, 0))
            output_test = np.clip(output_test, 0, 1)
            output_test = np.uint8(output_test * 255)
            output_test = cv2.cvtColor(output_test, cv2.COLOR_RGB2BGR)
            output_test = cv2.flip(output_test, 1)  # mirrors webcam frame
            cv2.imshow("frame", output_test)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    else:
        # Normal mode: just process an image.
        # Run the image through the model.
        image = load_image(INPUT_PATH)
        image = resize(image)
        output_test = model(image.to(device)).cpu().detach().numpy()
        save_image(OUTPUT_PATH, output_test[0, :, :, :])
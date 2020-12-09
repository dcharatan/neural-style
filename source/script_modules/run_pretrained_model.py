from source.image import load_image, save_image
import torch
from ..models.StylizationModel import StylizationModel
from PIL import Image
import torchvision.transforms as tf
import numpy as np
import cv2
import argparse

if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser(description="Neural Style Transfer")
    arg_parser.add_argument("-v", "--video", default=False, action="store_true",
                            help="toggle for style transfer on video stream")
    arg_parser.add_argument("-w", "--webcam", default=False, action="store_true",
                            help="toggle for style transfer on webcam stream")
    args = vars(arg_parser.parse_args())

    MODEL_PATH = "saved_models/final_model.pth"
    INPUT_PATH = "test/stadt.jpg"
    OUTPUT_PATH = "result/test1.jpg"
    IMAGE_SIZE = 256
    BATCH_SIZE = 4

    # Load the pre-trained model.
    model = StylizationModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    if args["video"]:
        pass

    if args["webcam"]: # webcam mode, process frames in webcam stream
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            test = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            test = load_image("", Image.fromarray(test))
            test = tf.Compose(
                [
                    tf.Resize(IMAGE_SIZE),
                    tf.CenterCrop(IMAGE_SIZE),
                ]
            )(test)
            output_test = model(test).detach().numpy()
            output_test = np.transpose(output_test[0, :, :, :], (1, 2, 0))
            output_test = np.clip(output_test, 0, 1)
            output_test = np.uint8(output_test * 255)
            output_test = cv2.cvtColor(output_test, cv2.COLOR_RGB2BGR)
            output_test = cv2.flip(output_test, 1) # mirrors webcam frame
            cv2.imshow('frame', output_test)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else: # normal mode, just process image
        # Run the image through the model.
        test = load_image(INPUT_PATH)
        test = tf.Compose(
            [
                tf.Resize(IMAGE_SIZE),
                tf.CenterCrop(IMAGE_SIZE),
            ]
        )(test)
        output_test = model(test).detach().numpy()
        save_image(OUTPUT_PATH, output_test[0, :, :, :])
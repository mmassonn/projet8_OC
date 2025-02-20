
import mimetypes
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import login, hf_hub_download
from flask import Flask, request, jsonify, send_file
from PIL import Image
import glob, os
import numpy as np
from matplotlib import colors

login(token="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")

REPO_ID = "mmassonn/CarSegmentation"
FILE_NAME = "mobilenet_unet_categorical_crossentropy_augFalse.keras"
model_file = hf_hub_download(repo_id=REPO_ID,filename=FILE_NAME)
MODEL = load_model(model_file, compile=False)


MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 128


def generate_img_from_mask(mask, colors_palette=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']):
    """Genère une image à partir du masque de segmentation."""

    id2category = {0: 'void',
                   1: 'flat',
                   2: 'construction',
                   3: 'object',
                   4: 'nature',
                   5: 'sky',
                   6: 'human',
                   7: 'vehicle'}

    img_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='float')

    for cat in id2category.keys():
        img_seg[:, :, 0] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[2]

    return img_seg


def predict_segmentation(image_array, image_width, image_height):
    '''Genère le masque de couleur à partir du modèle.'''

    image_array = Image.fromarray(image_array).resize((image_width, image_height))
    image_array = np.expand_dims(np.array(image_array), axis=0)
    mask_predict = MODEL.predict(image_array)
    mask_predict = np.squeeze(mask_predict, axis=0)
    mask_color = generate_img_from_mask(mask_predict) * 255

    return mask_color


app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome on the segmentation API"

@app.route("/predict_mask", methods=["POST"])
def segment_image():
    file = request.files['image']
    img = Image.open(file.stream)
    mask_color = predict_segmentation(image_array=np.array(img), image_width=MODEL_INPUT_WIDTH,
                                      image_height=MODEL_INPUT_HEIGHT)
    Image.fromarray(mask_color.astype(np.uint8)).save("tmp.png")
    return send_file("tmp.png", mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
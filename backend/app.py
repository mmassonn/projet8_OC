
import mimetypes
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import login, hf_hub_download
from huggingface_hub import HfApi, HfFolder
from flask import Flask, request, jsonify, send_file
from PIL import Image
import glob, os
import numpy as np
from matplotlib import colors
import base64
import io

login(token="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")

# Model
REPO_ID = "mmassonn/CarSegmentation"
MODEL_FILE_NAME = "mobilenet_unet_categorical_crossentropy_augFalse.keras"
model_file = hf_hub_download(repo_id=REPO_ID,filename=MODEL_FILE_NAME)
MODEL = load_model(model_file, compile=False)

# Dataset
DATASET_IMAGE_REPO_ID = "mmassonn/CarSegmentation_leftImg8bit"
DATASET_MASK_REPO_ID = "mmassonn/CarSegmentation_gtFine"

def get_dataset_file_path() -> tuple[list, dict]:
    """Extrais les chemins d'accès aux fichier dans dataset."""
    dataset_file_paths = {}
    api = HfApi()

    # Use list_repo_files instead of list_models_files
    image_files = api.list_repo_files(repo_id=DATASET_IMAGE_REPO_ID, repo_type="dataset")
    image_dataset_file_names = [file for file in image_files]

    mask_files = api.list_repo_files(repo_id=DATASET_MASK_REPO_ID, repo_type="dataset")
    mask_dataset_file_names = [file for file in mask_files]

    # Pair image and mask files
    for image_file, mask_file in zip(image_dataset_file_names, mask_dataset_file_names):
        dataset_file_paths[image_file] = mask_file

    return image_dataset_file_names, dataset_file_paths

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

@app.route("/image_path", methods=["POST"])
def get_image_file_path():
    image_dataset_file_names, _ = get_dataset_file_path()
    return jsonify(image_dataset_file_names[1:])

@app.route("/predict_mask", methods=['GET', 'POST'])
def segment_image() -> list:
    """Réalise la segmentation d'un images à partir de son path."""
    file = request.get_json(force=True)
    image_file_name = file['file_name']
    image_path = hf_hub_download(repo_id=DATASET_IMAGE_REPO_ID, filename=image_file_name, repo_type="dataset")
    image = Image.open(image_path)
    image_array = np.array(image)
    
    pred_mask_array = predict_segmentation(image_array=image_array, image_width=MODEL_INPUT_WIDTH,
                                            image_height=MODEL_INPUT_HEIGHT)
    
    _ , dataset_file_paths = get_dataset_file_path()
    real_mask_file_name = dataset_file_paths[image_file_name]
    real_mask_path = hf_hub_download(repo_id=DATASET_MASK_REPO_ID, filename=real_mask_file_name, repo_type="dataset")
    real_mask = Image.open(real_mask_path)
    real_mask_array = np.array(real_mask)

    # Convertir les arrays en images encodées en base64
    def array_to_base64(arr):
        img = Image.fromarray(arr.astype('uint8'))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        'image': array_to_base64(image_array),
        'real_mask': array_to_base64(real_mask_array),
        'pred_mask': array_to_base64(pred_mask_array)
    }

# if __name__ == "__main__":
#     # Launch the Flask app
#     app.run(debug=True)
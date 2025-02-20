import streamlit as st
import requests
from PIL import Image
from io import BytesIO

DATA_API_URL = "https://api.huggingface.com/your-endpoint"
PREDICTION_API_URL = "https://api.huggingface.com/your-endpoint"

def get_file_list_from_api():
    """Récupére de la liste des images disponibles depuis l'api."""
    data_api_url = DATA_API_URL
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération de la liste des fichiers.")
        return []

def send_post_request(file):
    """Envoie à l'api le nom de l'image sélectionnée."""
    url = PREDICTION_API_URL
    files = {'file': file}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de l'envoi de la requête POST.")
        return None

def display_images(image_urls):
    """Affiche l'image réelle, le masque segmentée réelle et celui réalisé par le modèle."""
    st.image(image_urls[0], caption="Image réelle")
    st.image(image_urls[1], caption="Segmentation réelle")
    st.image(image_urls[2], caption="Segmentation réalisée par le modèle")


st.title("Application de Segmentation d'Image")

file_list = get_file_list_from_api()
selected_file = st.selectbox("Sélectionnez un fichier", file_list)

if st.button("Création de l'image segmentée"):
    if selected_file:
        file_response = requests.get(str(selected_file))
        if file_response.status_code == 200:
            file = BytesIO(file_response.content)
            image_urls = send_post_request(file)
            if image_urls:
                display_images(image_urls)
        else:
            st.error("Erreur lors de la récupération du fichier sélectionné.")
    else:
        st.warning("Veuillez sélectionner un fichier.")

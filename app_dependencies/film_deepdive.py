from io import BytesIO
import requests
from PIL import Image


def get_actor_image(actor_path):
    root_img_path = 'https://image.tmdb.org/t/p/w500'
    url = root_img_path + actor_path
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def get_film_image(poster_url):
    response = requests.get(poster_url)
    img = Image.open(BytesIO(response.content))
    return img



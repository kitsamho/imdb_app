from io import BytesIO
import requests
from PIL import Image


def get_actor_frame(df_actor_all, actor):
    return df_actor_all[df_actor_all.actor == actor]


def get_actor_image_url(df_actor_specific_frame):
    root_img_path = 'https://image.tmdb.org/t/p/w500'
    url = root_img_path + df_actor_specific_frame.profile_path.unique()[0]
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img, url


def get_film_image(poster_url):
    response = requests.get(poster_url)
    img = Image.open(BytesIO(response.content))
    return img


def get_data_from_dic(df, col, key, string_return=True):
    strings = []
    for movie in df[col]:
        try:
            for keyword in movie:
                try:
                    strings.append(keyword[key])
                except:
                    pass
        except:
            pass

    if string_return:
        return ' '.join(strings)
    else:
        return [' '.join(i.split()) for i in strings]
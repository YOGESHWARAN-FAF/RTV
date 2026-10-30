import numpy as np
from PIL import Image, ImageFont, ImageDraw

def ImageCaption(image: np.array, caption: str, color=(255, 255, 255), font_size=40) -> np.array:
    h = image.shape[0]
    w = image.shape[1]
    img = Image.fromarray(image)
    font = ImageFont.truetype("./assets/fonts/Arial.ttf", font_size)
    draw = ImageDraw.Draw(img)

    draw.text((w // 2, h - h // 16), caption, color, font=font, anchor="ms")
    img = np.array(img)
    return img

def ImageCaptionBottom(image: np.array, caption: str, color=(255, 255, 255), font_size=40) -> np.array:
    h = image.shape[0]
    w = image.shape[1]
    img = Image.fromarray(image)
    font = ImageFont.truetype("./assets/fonts/Arial.ttf", font_size)
    draw = ImageDraw.Draw(img)

    draw.text((w // 2, h - h // 256), caption, color, font=font, anchor="ms")
    img = np.array(img)
    return img

def ImageCaptionBottomTNR(image: np.array, caption: str, color=(255, 255, 255), font_size=40) -> np.array:
    h = image.shape[0]
    w = image.shape[1]
    img = Image.fromarray(image)
    font = ImageFont.truetype("./assets/fonts/TimesNewRoman.ttf", font_size)
    draw = ImageDraw.Draw(img)

    draw.text((w // 2, h - h // 256), caption, color, font=font, anchor="ms")
    img = np.array(img)
    return img

def ImageCaptionTNR(image: np.array, caption: str, color=(255, 255, 255), font_size=40,x=0.5,y=0.5) -> np.array:
    h = image.shape[0]
    w = image.shape[1]
    img = Image.fromarray(image)
    font = ImageFont.truetype("./assets/fonts/TimesNewRoman.ttf", font_size)
    draw = ImageDraw.Draw(img)

    draw.text((int(w *x), int(h*y)), caption, color, font=font, anchor="ms")
    img = np.array(img)
    return img
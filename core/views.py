import os
import cv2
from PIL import Image
import numpy as np
import imutils

import tensorflow as tf
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError

from django.core.files.storage import FileSystemStorage


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name


def crop_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
              extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()

    return new_img


def index(request):
    IMG_SIZE = 256
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()
    try:
        image = request.FILES["image"]
        _image = fss.save(image.name, image)
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        tempPath = os.getcwd()+"/temp.png"
        image_url = fss.url(_image)
        imag = cv2.imread(path)
        imag = crop_img(imag)
        imag = cv2.resize(imag, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(tempPath, imag)
        imag = tf.keras.utils.load_img(path, target_size=(224, 224))
        input_arr = tf.keras.utils.img_to_array(imag) / 255
        input_arr = np.expand_dims(input_arr, axis=0)
        model = tf.keras.models.load_model(os.getcwd() + '/model.h5')

        prediction = (model.predict(input_arr) > 0.5).astype("int32")[0][0]

        os.remove(tempPath)

        if prediction == 1:
            prediction = "This MRI have brain tumor."
        else:
            prediction = "This MRI is of healthy brain."

        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "image": image,
                "image_url": image_url,
                "prediction": prediction,
            },
        )
    except MultiValueDictKeyError:

        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )

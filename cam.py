from multiprocessing import Pool
from itertools import starmap
import cv2
import numpy as np
import time
import torch as t

device = 'cpu'
vae = t.load('dumps/vae_dump', map_location=device)

# For webcam input:
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
#IMG_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
#IMG_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

success, image = cap.read()
# Vertical and horizontal indices

color = False
#color = True
while cap.isOpened():
    old_image = image
    success, image = cap.read()
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image.flags.writeable = False
    #cv2.imshow('Before', image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (vae.w, vae.h))/255

    ## First way
    #z = vae.encoder(t.tensor(image).float())
    #image_hat = vae.decoder(z)

    ## Second way
    image_hat = vae(t.tensor(image).float().reshape((1, 1, vae.w, vae.h)))
    image_hat = image_hat.detach().numpy()[0,0,:,:]

    img = np.concatenate((image, image_hat), axis=0)
    #img = image
    #img = image_hat
    #cv2.imshow('Figure', cv2.flip(img, 1))
    cv2.imshow('Figure', img)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


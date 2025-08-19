!pip -q install pyngrok > /dev/null
!pip -q install streamlit > /dev/null
!pip -q install patool > /dev/null

import cv2
import gdown
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import patoolib
import streamlit as st

from joblib import dump
from tqdm import tqdm
from pyngrok import ngrok

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import SGD

# Set up data directory
DATA_ROOT = '/content'
os.makedirs(DATA_ROOT, exist_ok=True)
!unzip "skin_cancer.zip" > /dev/null

# Get image paths from your skin_cancer folder
benign_img_paths = glob.glob('/content/skin_cancer/benign/*.jpg')
malignant_img_paths = glob.glob('/content/skin_cancer/malignant/*.jpg')

print(f"Found {len(benign_img_paths)} benign images")
print(f"Found {len(malignant_img_paths)} malignant images")

# Balance classes by using minimum count or cap at 3000
samples_per_class = min(len(benign_img_paths), len(malignant_img_paths), 3000)
total_samples = 2 * samples_per_class

print(f"Using {samples_per_class} samples per class")

# Make X and y from data
X = []
y = []

for i in tqdm(range(samples_per_class)):
  img = cv2.imread(benign_img_paths[i])
  X.append(cv2.resize(img, (224,224))) # standardize image size
  y.append(0)

for i in tqdm(range(samples_per_class)):
  img = cv2.imread(malignant_img_paths[i])
  X.append(cv2.resize(img, (224,224)))  # 224x224 instead of 50x50
  y.append(1)

X = np.array(X)
y = np.array(y)

print("Created our X and y variables")

# Save some sample images for trying out Streamlit app
skin_samples_dir = 'skin_samples'
if (os.path.exists(skin_samples_dir) == False):
  os.mkdir(skin_samples_dir)

for i, img in enumerate(X[samples_per_class-5:samples_per_class]):
  plt.imsave(f'benign_test_img_{i}.jpg', img)

for i, img in enumerate(X[samples_per_class:samples_per_class+5]):
  plt.imsave(f'malignant_test_img_{i}.jpg', img)

# Helpful function for launching our Streamlit app
def launch_website():
  print ("Click this link to try your web app:")
  if (ngrok.get_tunnels() != None):
    ngrok.kill()
  tunnel = ngrok.connect()
  print(tunnel.public_url)
  !streamlit run --server.port 80 app.py > /dev/null

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
plt.imshow(X[0])
plt.show()
print(f"Label: {y[0]}")

pca = PCA(n_components=20)

# Calculate the exact number automatically
total_elements = X.size  # Total elements in the array
num_images = X.shape[0]  # Number of images
pixels_per_image = total_elements // num_images  # Exact division

print(f"Total elements: {total_elements}")
print(f"Number of images: {num_images}")
print(f"Pixels per image: {pixels_per_image}")

X_pca = pca.fit_transform(np.reshape(X, (num_images, pixels_per_image)))

colors = ["green","brown"]
classes = [0,1]

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap=matplotlib.colors.ListedColormap(colors), s=5)
cb = plt.colorbar()
loc = np.arange(0,max(y),max(y)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(classes)
plt.title("PCA Representation")
plt.show()

X_gray = []

for i in range(total_samples):
  X_gray.append(cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY))

X_gray = np.array(X_gray)

plt.imshow(X_gray[0], cmap='gray')

X_gray_flat = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))

X_train_vgg, X_test_vgg, y_train_vgg, y_test_vgg = train_test_split(X, y, test_size=0.33, random_state=42)
# load the expert VGG16 network but do not include the final layers
vgg_expert = VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

# we add the first 12 layers of VGG16 to our own model vgg_model
vgg_model = Sequential()
vgg_model.add(vgg_expert)

# and then add our own layers on top of it
vgg_model.add(GlobalAveragePooling2D())
vgg_model.add(Dense(1024, activation = 'relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(512, activation = 'relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(1, activation = 'sigmoid'))

vgg_model.compile(loss = 'binary_crossentropy',
          optimizer = SGD(learning_rate=1e-4, momentum=0.95),
          metrics=['accuracy'])

vgg_model.fit(X_train_vgg, y_train_vgg,
              batch_size=64,
              epochs=30,
              verbose=1,
              validation_data=(X_test_vgg, y_test_vgg),
              shuffle=True)

print("Accuracy: {}".format(vgg_model.evaluate(X_test_vgg, y_test_vgg)[1], verbose=0))


from joblib import dump, load
dump(vgg_model, 'vgg_model.joblib')
clf = load('vgg_model.joblib')

!ngrok authtoken #ngroktoken

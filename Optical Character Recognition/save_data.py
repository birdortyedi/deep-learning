import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib.keras.python.keras.utils import to_categorical
import _pickle as cPickle
from tqdm import tqdm

letters = []
pixel_vals = []

print("Starting to read...")

with open('DATA/letter.data','r') as f:
    reader=csv.reader(f,delimiter='\t')
    for row in reader:
        letters.append(row[1])
        pixel_vals.append(row[6:-1])

print("Applying label encoding...")
labels = np.asarray(letters)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded,num_classes=26)

with open(r"lencoder.pkl", "wb") as output_file:
    cPickle.dump(le, output_file)

print("Creating collection...")
collection = np.zeros((1,20,20,1))
for i in tqdm(range(0,len(pixel_vals))):
    mat_img = np.asmatrix(pixel_vals[i]).reshape(16,8).astype(float)
    mat_img_tot = np.zeros((20,20))
    mat_img_tot[2:18,6:14] = mat_img
    collection = np.vstack((collection,mat_img_tot.reshape(1,20,20,1)))

collection = collection[1:]

print("Saving collection...")

np.save('collection.npy',collection)
np.save('labels_onehot.npy',labels_onehot)

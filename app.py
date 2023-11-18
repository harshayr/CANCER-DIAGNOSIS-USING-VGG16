import streamlit as st
import tensorflow as tf
from io import BytesIO, StringIO
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


MODEL = '/Users/harshalrajput/Desktop/MLOPS_cancer_project/artifacts/cancer_model.h5'

model = load_model(MODEL)

st.title( 'Lung Cancer Detector')

file = st.file_uploader('Upload File', type = ['PNG','CSV','JPEG'])
show_file = st.empty()

if not file:
    show_file.info('Upload image: {}'.format(['PNG', 'CSV', 'JPEG']))

# content = file.getvalue()


if isinstance(file, BytesIO):

    imp = image.load_img(file,target_size=(224, 224))
    # x = tf.cast(imp, tf.float64)
    x = image.img_to_array(imp)
    x = x/255
    x = np.expand_dims(x, axis=0)
    pred = np.argmax(model.predict(x), axis=1)
    if pred == 0:
        st.header('Adenocarcinoma Lung Cancer ' )
    elif pred == 1:
        st.header('Neuroendocrine Lung Cancer' )
    else:
        st.header('Squamous cell carcinoma Lung Cancer' )

    show_file.image(file)
# else:
#     show_file.info('Please upload valid file')






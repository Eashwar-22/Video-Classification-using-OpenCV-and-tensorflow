import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide",
                   page_title="Video Classification")


st.title("Video Classification - Deep Learning")
st.markdown("---")

st.markdown('''
            **Reference :** 
            - https://colab.research.google.com/drive/1RtTYonaJ7ASX_ZMzcV3t_0jNktheKQF9?usp=sharing
            - https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition
            - https://bleedai.com/human-activity-recognition-using-tensorflow-cnn-lstm
            ''')
st.markdown("---")
st.markdown('''**Model**''')
model_ = st.selectbox("Select Model",['ConvLSTM','None'])

show_arch = st.checkbox("Show Architecture")
show_code=st.checkbox("Show Code")
arch,img_arch= st.columns(2)
model_code,model_map = st.columns(2)

if model_=='ConvLSTM':
    if show_arch:
        with arch:
            st.subheader("Architecture")
            st.image('convlstm/convlstm_arch.png')
        with img_arch:
            st.subheader("Training per Video")
            st.image('convlstm/convlstm_imgarch.png')

    if show_code:
        st.subheader("Sample Model Code")
        st.code('''
        model = Sequential()
        model.add(ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True, input_shape = (SEQUENCE_LENGTH,
                                                                                          IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.2)))
        
        model.add(Flatten()) 
        
        model.add(Dense(len(CLASSES_LIST), activation = "softmax"))
        ''')
        st.subheader("Model Summary")
        im = Image.open('convlstm/convlstm_modelmap.png')
        st.image(im)




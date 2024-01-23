import numpy as np
import cv2
import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from model.NNModel import Network

# Loading the Model
model = Network() 
model.load_state_dict(torch.load("model/net.pth", map_location=torch.device("cpu"))) # Same Directory as Current Script - Load Trained Model

model.eval() # Model ready for Inference 

st.set_page_config(page_title="MLDA Project", page_icon=":tada:", layout = "wide")

with st.container():
    st.title("Using Convolutional Neural Network to predict Handwritten Digits")
    st.subheader("This is a Mini Project using the MNIST dataset to train a CNN Model and read hand written digits.")
    st.caption("Training Model can be found on GitHub")
    st.link_button(label='GitHub Link', url='https://github.com/tshjustin/MLDA-DecProj23')
st.divider()

text = st.text_area(label='Short Description:', value="Write any digit from 0-9 in the box and click Predict." 
                "The model would estimate the digit and output a corresponding label. The barchart shows the likehood of the numbers that was predicted by the Model.")

st.markdown('''Try writing a digit!''')
SIZE = 192

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

if canvas_result.image_data is not None and canvas_result.image_data.any():
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA) # display the resized image 
    st.write('Model Input')
    st.image(rescaled)

    if st.button('Predict'):
        #convert image from colour to gray
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #change the values from [0,255] to [0,1] then convert the numpy array to a torch tensor
        bwx = torch.from_numpy(test_x.reshape(1, 28, 28)/255)
        
        #change input to float from double then unsqueeze changes from (1,28,28) to (1,1,28,28)
        val = model(bwx.float().unsqueeze(0))
        
        st.write(f'Result: {np.argmax(val.detach().numpy()[0])}')
        st.bar_chart(np.exp(val.detach().numpy()[0]))
        print(np.exp(val.detach().numpy()[0]))

st.divider()

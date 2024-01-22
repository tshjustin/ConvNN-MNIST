import numpy as np
import cv2
import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from NNModel import CNN

model = CNN() # Instantiate the Model

model.load_state_dict(torch.load("deep_snn_model.pth", map_location=torch.device('cpu'))) # Same Directory as Current Script - Load Trained Model

model.eval() # Inference 

st.set_page_config(page_title="MLDA Project", page_icon=":tada:", layout = "wide")

with st.container():
    st.header("Justin's Project")
    st.title("Small Machine Learning Project using the MNSIT Dataset")


#these add text to the frontend
st.title('My MNIST model')
st.markdown('''Try to write a digit!''')


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


#process the image only if there is drawing
if canvas_result.image_data is not None and canvas_result.image_data.any():
    #resize the 192x192 canvas drawing to 28x28
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    #display the resized image (if you dont resize it back to 192x192, it'll show up as a small image)
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    st.write('Model Input')
    st.image(rescaled)

    #if predict button is pressed
    if st.button('Predict'):
        #convert image from colour to gray
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #change the values from [0,255] to [0,1] then convert the numpy array to a torch tensor
        bwx = torch.from_numpy(test_x.reshape(1, 28, 28)/255)
        #change input to float from double then unsqueeze changes from (1,28,28) to (1,1,28,28)
        val = model(bwx.float().unsqueeze(0))
        #result will be a one-hot tensor
        st.write(f'result: {np.argmax(val.detach().numpy()[0])}')
        #display the one-hot tensor output
        st.bar_chart(np.exp(val.detach().numpy()[0]))
        print(np.exp(val.detach().numpy()[0]))
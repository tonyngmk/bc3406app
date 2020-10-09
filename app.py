import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import tensorflow as tf
from tensorflow import keras
import numpy as np

st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/NTU%20Logo.png', width = 750)
st.write('''<h1 align=center><font color='Blue'>BC3406</font> - 
<font color='red'>Business Analytics Consulting</font>''', unsafe_allow_html=True)
# st.write("# Attila Cybertech - EDA")

use_case = st.sidebar.selectbox("Select use cases", 
("Exploratory Data Analysis",
"Detecting anomaly (encoder)",
"Predicting Single-Step (30 seconds)",
"Predicting Multi-Step (30 minutes)" 
))

# unsupervised_models = st.sidebar.selectbox("Select unsupervised model", 
# ("Convolutional Reconstruction Autoenconders - Pressure 1",
# "LSTM Reconstruction Autoenconders - Pressure 1",
# "Convolutional Reconstruction Autoenconders - Pressure 2",
# "LSTM Reconstruction Autoenconders - Pressure 2"))

@st.cache(persist = True)
def load_data():
    df = pd.read_csv("mainData_renamed.csv", nrows = 1000)
    df["time"] = pd.to_datetime(df["time"]) # convert string to datetime format
    df.reset_index(drop=True, inplace=True)
    return df
with st.spinner('Wait for 1000 rows to be loaded into memory...'):
    df = load_data()
# st.success('Done!')

class MyCallback(keras.callbacks.Callback):
  def on_predict_begin(self, logs=None):
    # keys = list(logs.keys())
    # print("Start predicting; got log keys: {}".format(keys))
    st.header('Percentage Complete')
    self._progress = st.empty()
    self._progress.progress(0)

  def on_predict_end(self, logs=None):
    # keys = list(logs.keys())
    # print("Stop predicting; got log keys: {}".format(keys))
    if batch % 100 == 99:
        self._summary_chart.add_rows(rows)
    batch_percent = logs['batch'] * logs['size'] / self.params['samples']
    percent = self._epoch / self._num_epochs + (batch_percent / self._num_epochs)
    self._progress.progress(math.ceil(percent * 100))

if use_case == "Exploratory Data Analysis":
    st.write("<h2 align=center>Attila Cybertech - EDA</h2>", unsafe_allow_html=True)
    startInput, endInput = st.select_slider(
        'Select a range of input',
        options=["Sensor {}".format(i) for i in range(1,  31)],
        value=("Sensor 5", "Sensor 10"))
    startInput = int(re.findall(r"Sensor (.*)", startInput)[0])
    endInput = int(re.findall(r"Sensor (.*)", endInput)[0])

    inputVars = st.multiselect(
        'Further filter input variables:',
        ["Sensor {}".format(i) for i in range(startInput,  endInput+1)],
        ["Sensor {}".format(i) for i in range(startInput,  endInput+1)])
    
    targetVars = st.multiselect(
        'Select target variables',
        ["Pressure {}".format(i) for i in range(1,  3)],
        ["Pressure {}".format(i) for i in range(1,  3)])

    if st.checkbox("Plot Correlation Matrix"):
        with st.spinner('Plotting matrix now...'):
            start_time = time.time()
            fig, ax = plt.subplots(figsize=(25, 25))
            sns.heatmap(df[inputVars + targetVars].corr(),cmap="BrBG",annot=True, fmt = 'g')
            "**Tip:** To observe graph better, click top-right, settings, wide mode."
            st.pyplot(fig)
        st.success('Done! Took {} seconds'.format(round(time.time()-start_time), 2))
        
    if st.checkbox("Plot Scatterplot Matrix"):
        with st.spinner('Plotting matrix now...'):
            start_time = time.time()
            fig, ax = plt.subplots(figsize=(25, 25))
            fig = sns.pairplot(df[inputVars + targetVars], diag_kind="kde")
            "**Tip:** To observe graph better, click top-right, settings, wide mode."
            st.pyplot(fig)
        st.success('Done! Took {} seconds'.format(round(time.time()-start_time), 2))
        
    if st.checkbox("Show Raw Data"):
        "**Sample dataset** (top 5 rows)"
        df[inputVars + targetVars].iloc[:6,:]
        "### Load Dataset"
        nrows = st.slider("Select number of rows  (up to 1000 to prevent high CPU usage)", 0, len(df))
        df[inputVars + targetVars].iloc[:nrows, :]


    # color = st.select_slider(
        # 'Select a color of the rainbow',
        # options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    # st.write('My favorite color is', color)
    # start_color, end_color = st.select_slider(
        # 'Select a range of color wavelength',
        # options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
        # value=('red', 'blue'))

    # number = st.number_input('Insert a number')
    # st.write('The current number is ', number)

elif use_case == "Detecting anomaly (encoder)":
    st.write("<h2 align=center>Attila Cybertech - Detecting anomaly (encoder)</h2>", unsafe_allow_html=True)
    targetVariable = st.sidebar.selectbox("Select target variable", 
    ("Pressure 1", "Pressure 2"))

    unsupervised_model = st.radio(
     "Which model are you interested in?",
    ('Convolutional', 'LSTM'))

    # TIME_STEPS = 120
    # training_mean = df["Pressure 1"].mean()
    # training_std = df["Pressure 1"].std()
    # df_training_value = (df["Pressure 1"] - training_mean) / training_std
    # def create_sequences(values, time_steps=TIME_STEPS):
    #     output = []
    #     for i in range(len(values) - time_steps):
    #         output.append(values[i : (i + time_steps)])
    #     return np.expand_dims(np.stack(output), axis = 2)
    # X_train = create_sequences(df_training_value.values)

    if unsupervised_model == "Convolutional" and targetVariable == "Pressure 1":
      # with st.spinner('Loading Convolutional Autoencoder model for Pressure 1 from Google Cloud Storage...'):
      #   start_time = time.time()
      #   model = keras.models.load_model("gs://tonyng/4.1.1.1_autoencoder_pressure1")
      # st.success('Model loaded! Took {} seconds'.format(round(time.time()-start_time), 2))
      # with st.spinner('Waiting for model to regenerate input...'):
      #   start_time = time.time()
      #   X_train_pred = model.predict(X_train, verbose = 1, callbacks=[MyCallback()])
      # st.success('Prediction obtained! Took {} seconds'.format(round(time.time()-start_time), 2))
      # train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
      # fig = plt.figure(figsize = (20, 10))
      # plt.hist(train_mae_loss, bins=50)
      # plt.xlabel("Train MAE loss", fontsize=14)
      # plt.ylabel("No of samples", fontsize=14)
      # plt.tick_params(axis="x", labelsize=12)
      # plt.tick_params(axis="y", labelsize=12)
      # plt.show()
      # st.pyplot(fig)
      # # Get reconstruction loss threshold.
      # threshold = np.max(train_mae_loss)
      # print("Reconstruction error threshold: ", threshold)
      "### Convolutional Reconstruction Model - `Pressure 1`"
      '''*Using cached images as actual prediction takes over 24 hours while running on CPU.*'''
      '**Tip:** To observe graph better, click top-right, settings, wide mode.'

      "#### Training and Validation Loss"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convLossPressure1.png', width = 750)
      "#### Train MAE Loss over sequences"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convMaxLossPressure1.png', width = 750)
      "Max Train MAE Loss: 0.042929180939266315"
      "#### Reconstruction with Autoencoder"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convPredPressure1.png', width = 750)
      '''#### Anomalies Across Dataset
      Threshold: 0.02
      '''
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convAnomaliesPressure1.png', width = 750)
      "#### Anomaly Samples"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convAnomaliesSamplePressure1.png', width = 750)

    elif unsupervised_model == "LSTM" and targetVariable == "Pressure 1":
      "### LSTM Reconstruction Model - `Pressure 1`"
      '''*Using cached images as actual prediction takes over 24 hours while running on CPU.*'''
      '**Tip:** To observe graph better, click top-right, settings, wide mode.'

      "#### Training and Validation Loss"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmLossPressure1.png', width = 750)
      "#### Train MAE Loss over sequences"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmMaxLossPressure1.png', width = 750)
      "Max Train MAE Loss: 265.5975389708013"
      "#### Reconstruction with Autoencoder"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmPredPressure1.png', width = 750)
      '''#### Anomalies Across Dataset
      Threshold: 3
      '''
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmAnomaliesPressure1.png', width = 750)
      "#### Anomaly Samples"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmAnomaliesSamplePressure1.png', width = 750)

    elif unsupervised_model == "Convolutional" and targetVariable == "Pressure 2":
      "### Convolutional Reconstruction Model - `Pressure 2`"
      '''*Using cached images as actual prediction takes over 24 hours while running on CPU.*'''
      '**Tip:** To observe graph better, click top-right, settings, wide mode.'

      "#### Training and Validation Loss"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convLossPressure2.png', width = 750)
      "#### Train MAE Loss over sequences"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convMaxLossPressure2.png', width = 750)
      "Max Train MAE Loss: 0.16287028699666112"
      "#### Reconstruction with Autoencoder"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convPredPressure2.png', width = 750)
      '''#### Anomalies Across Dataset
      Threshold: 0.02
      '''
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convAnomaliesPressure2.png', width = 750)
      "#### Anomaly Samples"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convAnomaliesSamplePressure2.png', width = 750)

    elif unsupervised_model == "LSTM" and targetVariable == "Pressure 2":
      "### LSTM Reconstruction Model - `Pressure 2`"
      '''*Using cached images as actual prediction takes over 24 hours while running on CPU.*'''
      '**Tip:** To observe graph better, click top-right, settings, wide mode.'

      "#### Training and Validation Loss"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmLossPressure2.png', width = 750)
      "#### Train MAE Loss over sequences"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmMaxLossPressure2.png', width = 750)
      "Max Train MAE Loss: 108.13926290197575"
      "#### Reconstruction with Autoencoder"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmPredPressure2.png', width = 750)
      '''#### Anomalies Across Dataset
      Threshold: 3
      '''
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/convAnomaliesPressure2.png', width = 750)
      "#### Anomaly Samples"
      st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/lstmAnomaliesSamplePressure2.png', width = 750)

elif use_case == "Predicting Single-Step (30 seconds)":
    st.write("<h2 align=center>Attila Cybertech - Predicting Single-Step (30 seconds)</h2>", unsafe_allow_html=True)
    targetVariable = st.sidebar.selectbox("Select target variable", 
    ("Pressure 1", "Pressure 2"))

    singleStepModel = st.radio(
         "Which model are you interested in?",
        ('Linear', 'Dense', 'Convolutional', 'Recurrent (LSTM)', "CNN-LSTM", 'Summary'))

    if targetVariable == "Pressure 1":
      "### Single-Step Prediction Model - `Pressure 1`"
      if singleStepModel == "Linear":
        "#### Linear Diagram"
        # st.image("https://www.tensorflow.org/tutorials/structured_data/images/wide_window.png", width = 500)
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearPressure1.png', width = 750)

        st.subheader("Results")
        # df = pd.DataFrame({"Validation Loss": 0.0671, "Test Loss": 0.0725})
        df = pd.DataFrame({"Loss": [0.0671, 0.0725]},index=['Validation Loss', 'Test Loss'])
        df
        # """
        # - Validation Loss: 0.0671
        # - Test Loss: 0.0725
         # """
      elif singleStepModel == "Dense":
        "#### Dense Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDenseDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDensePressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDensePressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDensePressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.0295, 0.0523]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "Convolutional":
        "#### Convolutional Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDenseDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleConvPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleConvPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleConvPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.1352, 0.1368]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "Recurrent (LSTM)":
        "#### RNN (LSTM) Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.0041, 0.0116]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "CNN-LSTM":
        "#### CNN-LSTM Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDenseDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleCnnRnnPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleCnnRnnPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleCnnRnnPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.0624, 0.1042]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "Summary":
        "#### Validation MAE"
        '''
        ```
        Dense       : 0.0295
        LSTM        : 0.0041
        Convolution : 0.1352
        CNN-LSTM    : 0.0624
        Linear      : 0.0671
        '''
        "#### Test MAE"
        '''
        ```
        Dense       : 0.0523
        LSTM        : 0.0116
        Convolution : 0.1368
        CNN-LSTM    : 0.1042
        Linear      : 0.0725
        '''         
        "#### Summary MAE"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/summaryPressure1.png', width = 700)
        st.subheader("Conclusion")
        "**LSTM** is the superior model for single-step prediction for `Pressure 1`."

    elif targetVariable == "Pressure 2":
      "### Single-Step Prediction Model - `Pressure 2`"
      if singleStepModel == "Linear":
        "#### Linear Diagram"
        # st.image("https://www.tensorflow.org/tutorials/structured_data/images/wide_window.png", width = 500)
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        # st.subheader("Model architecture")
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearPressure2Model.png", width = 400)
        # st.subheader("Hyperparameters")
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        # st.subheader("Training and Validation loss")
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.1894, 0.1592]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "Dense":
        "#### Dense Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleLinearDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDensePressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDensePressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDensePressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.0929, 0.1237]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "Convolutional":
        "#### Convolutional Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDenseDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleConvPressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleConvPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleConvPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.1310, 0.3550]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "Recurrent (LSTM)":
        "#### RNN (LSTM) Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnPressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleRnnPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.0633, 0.0827]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "CNN-LSTM":
        "#### CNN-LSTM Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleDenseDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleCnnRnnPressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleCnnRnnPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singleCnnRnnPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.1153, 0.1896]},index=['Validation Loss', 'Test Loss'])
        df
      elif singleStepModel == "Summary":
        "#### Validation MAE"
        '''
        ```
        Linear      : 0.1894
        Dense       : 0.0929
        Convolution : 0.1310
        LSTM        : 0.0633
        CNN-LSTM    : 0.1153
        '''
        "#### Test MAE"
        '''
        ```
        Linear      : 0.1592
        Dense       : 0.1237
        Convolution : 0.3550
        LSTM        : 0.0827
        CNN-LSTM    : 0.1896
        '''         
        "#### Summary MAE"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/singlePredictPressure2.png', width = 700)
        st.subheader("Conclusion")
        "**LSTM** is the superior model for single-step prediction for `Pressure 2`."         
         
elif use_case == "Predicting Multi-Step (30 minutes)":
    st.write("<h2 align=center>Attila Cybertech - Predicting Multi-Step (30 minutes)</h2>", unsafe_allow_html=True)
    targetVariable = st.sidebar.selectbox("Select target variable", 
    ("Pressure 1", "Pressure 2"))

    multiStepModel = st.radio(
         "Which model are you interested in?",
        ('Last', 'Repeat', 'Linear', 'Dense', 'Convolutional', 'Recurrent (LSTM)', "CNN-LSTM", "Autoregressive (LSTM)", 'Summary'))

    if targetVariable == "Pressure 1":
      "### Multi-Step Prediction Model - `Pressure 1`"
      if multiStepModel == "Last":
        "#### Last Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLastDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLastPressure1Model.png", width = 200)
        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLastPressure1.png', width = 750)
        st.subheader("Results")
        df = pd.DataFrame({"Loss": [1.0430, 0.8980]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Repeat":
        "#### Repeat Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRepeatDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRepeatPressure1Model.png", width = 125)
        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRepeatPressure1.png', width = 750)
        st.subheader("Results")
        df = pd.DataFrame({"Loss": [1.0447, 0.8983]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Linear":
        "#### Linear Diagram"
        # st.image("https://www.tensorflow.org/tutorials/structured_data/images/wide_window.png", width = 500)
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        # st.subheader("Model architecture")
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.4414, 0.3299]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Dense":
        "#### Dense Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiDensePressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiDensePressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiDensePressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.3681, 0.4306]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Convolutional":
        "#### Convolutional Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiConvPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiConvPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiConvPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.2930, 0.3689]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Recurrent (LSTM)":
        "#### RNN (LSTM) Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.2323, 0.3580]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "CNN-LSTM":
        "#### CNN-LSTM Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnRnnPressure1Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnRnnPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnRnnPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.3505, 0.3937]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Autoregressive (LSTM)":
        "#### Autoregressive (LSTM) Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmDiagram.png", width = 700)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        # "#### Model architecture"
        # st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmPressure1Model.png", width = 125)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmPressure1Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmPressure1.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.2113, 0.2653]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Summary":
        "#### Validation MAE"
        '''
        ```
        Last        : 1.0430
        Repeat      : 1.0447
        Linear      : 0.4414
        Dense       : 0.3681
        Conv        : 0.2930
        LSTM        : 0.2323
        CNN-LSTM    : 0.3505
        AR-LSTM     : 0.2113
        '''
        "#### Test MAE"
        '''
        ```
        Last        : 0.8980
        Repeat      : 0.8983
        Linear      : 0.3299
        Dense       : 0.4306
        Conv        : 0.3689
        LSTM        : 0.3580
        CNN-LSTM    : 0.3937
        AR-LSTM     : 0.2653
        '''         
        "#### Summary MAE"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiStepPredictPressure1(3).png', width = 700)
        st.subheader("Conclusion")
        "**AR-LSTM** is the superior model for multiple-step prediction for `Pressure 1`."         

    elif targetVariable == "Pressure 2":
      "### Multi-Step Prediction Model - `Pressure 2`"
      if multiStepModel == "Last":
        "#### Last Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLastDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLastPressure2Model.png", width = 200)
        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLastPressure2.png', width = 750)
        st.subheader("Results")
        df = pd.DataFrame({"Loss": [1.0433, 0.8982]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Repeat":
        "#### Repeat Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRepeatDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRepeatPressure2Model.png", width = 200)
        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRepeatPressure2.png', width = 750)
        st.subheader("Results")
        df = pd.DataFrame({"Loss": [1.0448, 0.8983]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Linear":
        "#### Linear Diagram"
        # st.image("https://www.tensorflow.org/tutorials/structured_data/images/wide_window.png", width = 500)
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        # st.subheader("Model architecture")
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearPressure2Model.png", width = 400)
        # st.subheader("Hyperparameters")
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        # st.subheader("Training and Validation loss")
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.4484, 0.3336]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Dense":
        "#### Dense Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiLinearDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiDensePressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiDensePressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiDensePressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.3661, 0.4495]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Convolutional":
        "#### Convolutional Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiConvPressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiConvPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiConvPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.3063, 0.3667]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Recurrent (LSTM)":
        "#### RNN (LSTM) Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnPressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.2487, 0.3044]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "CNN-LSTM":
        "#### CNN-LSTM Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiRnnDiagram.png", width = 415)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        "#### Model architecture"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnRnnPressure2Model.png", width = 400)
        "#### Other hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnRnnPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiCnnRnnPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.3509, 0.4097]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Autoregressive (LSTM)":
        "#### Autoregressive (LSTM) Diagram"
        st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmDiagram.png", width = 700)
        '''*Using cached images as actual prediction takes a long time while running on CPU.*'''
        # "#### Model architecture"
        # st.image("https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmPressure2Model.png", width = 125)
        "#### OTher hyperparameters"
        df = pd.DataFrame({"Value": [512, "Patience 2 on validation loss"]},index=['Batch size', 'Early-stopping'])
        df
        "#### Training and Validation loss"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmPressure2Loss.png', width = 750)

        "#### Input & Predictions plot"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiArLstmPressure2.png', width = 750)

        st.subheader("Results")
        df = pd.DataFrame({"Loss": [0.2136, 0.3052]},index=['Validation Loss', 'Test Loss'])
        df
      elif multiStepModel == "Summary":
        "#### Validation MAE"
        '''
        ```
        Last        : 1.0433
        Repeat      : 1.0448
        Linear      : 0.4484
        Dense       : 0.3661
        Conv        : 0.3063
        LSTM        : 0.2487
        CNN-LSTM    : 0.3509
        AR-LSTM     : 0.2136
        '''
        "#### Test MAE"
        '''
        ```
        Last        : 0.8982
        Repeat      : 0.8983
        Linear      : 0.3336
        Dense       : 0.4495
        Conv        : 0.3667
        LSTM        : 0.3044
        CNN-LSTM    : 0.4097
        AR-LSTM     : 0.3052
        '''         
        "#### Summary MAE"
        st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/streamlit/Atilla/multiStepPredictPressure2(2).png', width = 700)
        st.subheader("Conclusion")
        "**AR-LSTM** is the superior model for multiple-step prediction for `Pressure 2`."         



else:
    '''
    ### 1. Unsupervised Learning
    '''

    '''
    ### 2. Supervised Models
    '''
    
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

myWatermark = """
            <style>
            footer:after {
            content:'Tony Ng'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            }
            </style>
            """
st.markdown(myWatermark, unsafe_allow_html=True)
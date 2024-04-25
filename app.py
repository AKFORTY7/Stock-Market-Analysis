import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model # type: ignore
import streamlit as st
import matplotlib.pyplot as plt

class SimpleChatBot:
    def __init__(self):
        self.responses = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hi there! What can I help you with?",
            "how are you": "I'm just a bot, but I'm here to help you!",
            "bye": "Goodbye! Have a great day!",
        }

    def respond(self, message):
        message = message.lower()
        if message in self.responses:
            return self.responses[message]
        else:
            return "I'm sorry, I didn't understand that. Can you please rephrase?"
# Initialize the chatbot
chatbot = SimpleChatBot()

model = load_model('C:/Users/Kalidas/OneDrive/Desktop/LLM/Stock Predictions Model.keras')

st.header('Stock Market Analysis')

# Function to handle user input and display chatbot responses
def chat():
    user_input = st.text_input("You:", "")
    if user_input:
        response = chatbot.respond(user_input)
        st.text_area("Bot:", value=response, height=100)

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start ,end)
st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

# Display the chat interface
st.subheader("Chat with the Bot")
chat()

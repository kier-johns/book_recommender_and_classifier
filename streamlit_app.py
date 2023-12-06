#Imports
import pickle
import streamlit as st

st.title('Will This Book Save You From Your Book Hangover?!')

st.write('testing')

#Load pickled pipeline so we can take inouts and make predictions
with open('../code/book_pipe.pkl', 'rb') as pickle_in:
    pipe = pickle.load(pickle_in)

# get user input
user_text = st.text_input('Please input some text: ')

#now generate predictions
predicted = pipe.predict([user_text])[0]

#display predictions
st.write(f'Your prediction is: {predicted}')
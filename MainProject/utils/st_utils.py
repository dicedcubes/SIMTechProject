"""
This file contains all the utility functions for Streamlit.
display_header_and_image    Displays the header information for the chatbot.
initialize_session          Initializes the session variables, such as responses, requests, etc.
display_chat_history        Displays the chat history.
"""

# Imports
import streamlit as st


def display_header_and_image():
    """
    Displays the header information for the chatbot and an image.
    """

    st.markdown('# Data Analysis Chatbot')
    st.markdown('Powered by Langchain Agents, SKLearn, and Streamlit')

    # # For Side Images
    # image = Image.open('images/ref.png')
    # width, height = image.size
    # image = image.resize((width // 2, height // 2))
    # st.sidebar.image(image, caption='Image created by DALL·E 3')

def initialize_session():
    """
    Initializes or resets session variables.
    """
    if 'responses' not in st.session_state:
        st.session_state['responses'] = [{'text': 'How can I assist you?', 'image_url': None}]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if 'side_images' not in st.session_state:
        st.session_state['side_images'] = []
    if 'input_query' not in st.session_state:
        st.session_state['input_query'] = ''

def display_chat_history():
    """
    Displays the chat history.
    """
    for i, response in enumerate(st.session_state['responses']):
        with st.chat_message('assistant'):
            st.write(response['text'])
            if response['image_url']:
                st.image(response['image_url'], use_column_width=True)

        if i < len(st.session_state['requests']):
            with st.chat_message('user'):
                st.write(st.session_state['requests'][i])

def display_side_images():
    """
    Displays images and graphs in the sidebar.
    """
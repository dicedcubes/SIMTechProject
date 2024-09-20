# Imports
import streamlit as st
from PIL import Image
import re

from utils import agent, st_utils

# Constants
MAX_MESSAGES = 20

def main():
    st_utils.display_header_and_image()
    st_utils.initialize_session()

    if 'agent' not in st.session_state:
        st.session_state.agent = agent.create_agent(max_msgs=MAX_MESSAGES)

    # Container for chat history
    chat_container = st.container()

    # Container for user's prompt
    prompt_container = st.container()

    # Sidebar for images and graphs
    # image_sidebar = st.sidebar()

    with prompt_container:
        def submit():
            st.session_state.input_query = st.session_state.widget
            st.session_state.widget = ''

        st.text_input('Prompt: ', placeholder='Enter your prompt here.', key='widget', on_change=submit)

        query = st.session_state.input_query

        if query:
            with st.spinner('Generating Response...'):
                # Store the query in the session state
                st.session_state.requests.append(query)

                # Query the LLM for response
                result = st.session_state.agent({'input': query})

                # Extract the URL from the result
                pattern = r'(.*:)\s*\[.*?\]\((.*?)\)'
                match = re.search(pattern, result['output'])
                if match:
                    text_before_link = match.group(1)
                    image_url = match.group(2)

                # else:
                text_before_link = result['output']
                image_url = None

                # Store the response in the session state
                st.session_state.responses.append({
                    'text': text_before_link,
                    'image_url': image_url,
                })

    with chat_container:
        st_utils.display_chat_history()

    # with image_sidebar:
    #     pass

if __name__ == '__main__':
    main()
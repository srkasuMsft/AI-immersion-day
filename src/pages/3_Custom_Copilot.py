import streamlit as st
import urllib.request
import json
import os
import ssl
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()
AZURE_ENDPOINT_KEY = os.environ['AZURE_ENDPOINT_KEY'] = 'ZhFEuCN6k2CWgIG8OG6xMbDfqbDDhGEB'

def allowSelfSignedHttps(allowed):
    # Bypass the server certificate verification on the client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
# Streamlit UI components
    #st.image("esg.png", width=600)
    st.title(' Welcome to your Essential of ESG Assistant!')
    st.sidebar.title(" Copilot for comparing MSFT and GOOG ESG initiatives for the years 2022 thru 2024 !")
    st.sidebar.caption("Made by an Srini Kasu")
    st.sidebar.info("""
    Generative AI technology has the potential to greatly enhance our understanding of ESG efforts of these Mega Corportations, particularly in fields like carbon reduction and adopting alternate energy. 
    This is because AI platforms can quickly summarize and provide highly detailed and interactive understand of Megacap corporations are doing to reduce the carbon footprint.
    """)
def main():
    allowSelfSignedHttps(True)
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for interaction in st.session_state.chat_history:
        if interaction["inputs"]["chat_input"]:
            with st.chat_message("user"):
                st.write(interaction["inputs"]["chat_input"])
        if interaction["outputs"]["chat_output"]:
            with st.chat_message("assistant"):
                st.write(interaction["outputs"]["chat_output"])

    # React to user input
    if user_input := st.chat_input("Ask me anything..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)

        # Query API
        data = {"chat_history": st.session_state.chat_history, 'chat_input': user_input}
        body = json.dumps(data).encode('utf-8')
        url = 'https://bokf-prj-endpoint.eastus.inference.ml.azure.com/score'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': ('Bearer '+ AZURE_ENDPOINT_KEY),
            'azureml-model-deployment': 'bokf-prj-endpoint-1'
        }
        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)
            response_data = json.loads(response.read().decode('utf-8'))

            # Check if 'chat_output' key exists in the response_data
            if 'chat_output' in response_data:
                with st.chat_message("assistant"):
                    st.markdown(response_data['chat_output'])

                st.session_state.chat_history.append(
                    {"inputs": {"chat_input": user_input},
                     "outputs": {"chat_output": response_data['chat_output']}}
                )

            else:
                st.error("The response data does not contain a 'chat_output' key.")
        except urllib.error.HTTPError as error:
            st.error(f"The request failed with status code: {error.code}")

if __name__ == "__main__":
    main()
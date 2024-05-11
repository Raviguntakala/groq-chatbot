import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'responce' not in st.session_state:
    st.session_state.responce = []
if 'question' not in st.session_state:
    st.session_state.question = []

def main():

    st.title("Personal AI Assistant")
    st.subheader("Your AI-powered copilot for queries:")
    USER_AVATAR = "https://api.dicebear.com/6.x/adventurer/svg?"
    ASSISTANT_AVATAR = "https://ask.vanna.ai/static/img/vanna_circle.png"
    # Add customization options to the sidebar
    st.sidebar.title('Select an LLM')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096','llama3-70b-8192','llama3-8b-8192','gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)
    temperature_length = st.sidebar.slider('Temperature:', 0.0, 1.0, value = 0.5,step=0.1)
    Max_Tokens_length = st.sidebar.slider('Max Tokens:', 0, 32768, value = 1024,step=1024)

    user_question = st.chat_input('Enter your question here...' , key="message")

    # session state variable
    if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=conversational_memory_length,return_messages=True)
    system_prompt = 'You are a helpful AI Personal Assistant'
    system_msg_template = SystemMessagePromptTemplate.from_template(template=system_prompt)
 
 
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    for messages in st.session_state.messages:
        avatar_url = ASSISTANT_AVATAR if messages["role"] == "assistant" or messages["role"] == "str" else None #USER_AVATAR 
        with st.chat_message(messages["role"], avatar=avatar_url):
            st.write(messages["content"])   # Keep user  messages in markdown container

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            # The language model which will generate the responses.
            model_name=model,
            # Controls randomness: lowering results in less random responses.
            # As the temperature approaches zero, the model will become deterministic
            # and repetitive.
            temperature=temperature_length,

            # The maximum number of tokens to generate. Requests can use up to
            # 32,768 tokens shared between prompt and completion.
            max_tokens=Max_Tokens_length,

            # Controls diversity via nucleus sampling: 0.5 means half of all
            # likelihood-weighted options are considered.
            top_p=1,

            # A stop sequence is a predefined or user-specified text string that
            # signals an AI to stop generating content, ensuring its responses
            # remain focused and concise. Examples include punctuation marks and
            # markers like "[end]".
            stop=None

    )

    conversation = ConversationChain(
            llm=groq_chat,
            memory=st.session_state.buffer_memory,
            prompt=prompt_template,
            verbose=True
    )
    
    if user_question:
        st.chat_message("user").write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        response = conversation(user_question)
        st.chat_message("assistant",avatar=ASSISTANT_AVATAR).write(response['response'])
        st.session_state.messages.append({"role": "assistant", "content": response['response']}) 
        # message = {'human':user_question,'AI':response['response']}
        # st.session_state.chat_history.append(message)
        # st.write("Chatbot:", response['response'])

if __name__ == "__main__":
    main()
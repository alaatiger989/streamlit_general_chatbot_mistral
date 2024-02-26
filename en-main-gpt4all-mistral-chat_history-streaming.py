from deep_translator import GoogleTranslator
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download

from langchain_community.llms import LlamaCpp
import gpt4all
from gpt4all import GPT4All
from deep_translator import GoogleTranslator
import asyncio
import sys
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage


@st.cache_resource
def create_chain():
    model = GPT4All(device = 'gpu' , model_name = "alaa_ai_model_mistral_v1.9.gguf" , model_path ='/home/administrator/' , allow_download = False , n_ctx = 2048 , n_threads =5 , ngl = 5)
    #st.markdown("Model working on " + str(model.n_threads) + " Threads , and " + str(model.ngl) + " gpu Layers " )
    st.markdown("Model Loaded Successfully....")
    return model


# Set the webpage title
st.set_page_config(
    page_title="Alaa's Chat Robot!"
)

# Create a header element
st.header("Alaa's Chat Robot!")



# Create Select Box
lang_opts = ["ar", "en" , "fr" , "zh-CN"]
lang_selected = st.selectbox("Select Target Language " , options = lang_opts)
# This sets the LLM's personality for each prompt.
# The initial personality provided is basic.
# Try something interesting and notice how the LLM responses are affected.
#system_prompt = st.text_area(
    #label="System Prompt",
    #value="You are a helpful AI assistant who answers questions in short sentences.",
    #key="system_prompt")


# Create LLM chain to use for our chatbot.
mod = create_chain()

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def disable():
    st.session_state.disabled = True
    
if "disabled" not in st.session_state:
    st.session_state.disabled = False

async def get_response(mod , user_prompt , container):
    full_response = ""
    generator = mod.generate(prompt = user_prompt , max_tokens = 1000 , streaming = True)
    event_loop = asyncio.get_running_loop()
    has_tokens = True
    def consume(generator):
        nonlocal has_tokens
        try:
            return next(generator)
        except:
            has_tokens = False
    # Add the response to the chat window
    with container.chat_message("assistant"): 
        while has_tokens:
            token = await event_loop.run_in_executor(None , consume , generator)
            if token is not None:
                full_response+=token
                container.markdown(full_response)
    full_response = GoogleTranslator(source='auto', target=lang_selected).translate(full_response)
    container.markdown(full_response)
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

with mod.chat_session():
# We take questions/instructions from the chat input to pass to the LLM
    if user_prompt := st.chat_input("Your message here", key="user_input" , on_submit = disable , disabled=st.session_state.disabled):
        del st.session_state.disabled
        if "disabled" not in st.session_state:
            st.session_state.disabled = False
        #st.chat_input("Your message here", key="disabled_chat_input", disabled=True)
        st.markdown("in session")
        # Add our input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )
        # Add our input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        
        user_prompt = GoogleTranslator(source='auto', target='en').translate(user_prompt)
        asyncio.run(get_response(mod , user_prompt , st.empty()))
        
##
        
        st.rerun()


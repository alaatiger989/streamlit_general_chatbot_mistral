from deep_translator import GoogleTranslator
import streamlit as st
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
import os
import asyncio
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



@st.cache_resource
def create_chain():
    # embedding engine
    hf_embedding = HuggingFaceInstructEmbeddings()
    # Initialize StreamingHandler Callback Manager
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Loading Model
    llm = LlamaCpp(
    # model_path="llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_path="/home/administrator/alaa_ai_model_llama2_v4_M.gguf",
        n_ctx=2000,
        n_gpu_layers=512,
        n_batch=30,
        callback_manager=callback_manager,
        temperature = 0.1,
        max_tokens = 30000,
        n_parts=1,
    )
    
    # Prompt Template
    template = """Question: {question}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    # Init Chain Rule
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    # Checking the Knowledge Base
    filePath = '/home/administrator/faiss_knowledge_base/'
    # Reading IP Address Json file of User
    #if(os.path.isfile(filePath)):
    st.markdown("Knowledge Base is loading...")
    # load from local
    db = FAISS.load_local("faiss_knowledge_base/", embeddings=hf_embedding)
    st.markdown("Knowledge Base Loaded Successfully ...")
    #else:
        #st.markdown("Books are loading ...")
        #knowledge_base_loader = PyPDFDirectoryLoader("pdfs")
        #knowledge_base = knowledge_base_loader.load()
        #knowledge_base_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        #knowledge_base_texts = knowledge_base_text_splitter.split_documents(knowledge_base)
        #st.markdown("Knowledge Base For " + str(len(knowledge_base_texts)) + " Pages ")
        #db = FAISS.from_documents(knowledge_base_texts, hf_embedding)

        # save embeddings in local directory
        #db.save_local("faiss_knowledge_base")
        #st.markdown("Knowledge Base Saved Successfully...")
    return db , llm_chain , prompt


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
system_prompt = st.text_area(
    label="System Prompt",
    value="Your name is Abdallah Fawzy . You are an expert in Egyptian Law and helpful AI assistant who answers questions in Details. You have all access to solve cases and provide intelligence solutions.",
    key="system_prompt")


# Create LLM chain to use for our chatbot.
db , llm_chain , prompt = create_chain()
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

async def get_response(db , llm_chain , prompt ,user_prompt , container):
    full_response = ""
    search = db.similarity_search(user_prompt, k=2)
    #st.markdown(search)
    template = '''Context: {context}

    Based on Context provide me answer for following question
    Question: {question}

    Your name is Abdallah Fawzy . You are an expert in Egyptian Law and helpful experienced Lawyer who answers questions in Details and organized manner. You have all access to solve cases and provide intelligence solutions. Tell me the information about the fact. The answer should be from context
    use general knowledge to answer the query'''
    prompt = PromptTemplate(input_variables=["context", "question"], template= template)
    final_prompt = prompt.format(question=user_prompt, context=search)
    with container.chat_message("assistant"):
        full_response = llm_chain.run(final_prompt)
        container.markdown(full_response)
    full_response = GoogleTranslator(source='auto', target=lang_selected).translate(full_response)
    container.markdown(full_response)
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
if user_prompt := st.chat_input("Your message here", key="user_input" , on_submit = disable , disabled=st.session_state.disabled):
        del st.session_state.disabled
        if "disabled" not in st.session_state:
            st.session_state.disabled = False
        # Add our input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )
        # Add our input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        
        user_prompt = GoogleTranslator(source='auto', target='en').translate(user_prompt)
        asyncio.run(get_response(db , llm_chain , prompt , user_prompt , st.empty()))
        
##
        
        st.rerun()

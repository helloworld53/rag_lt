import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from pinecone import Pinecone
from huggingface_hub import hf_hub_download
@st.cache_resource()
def load_model():

    # from google.colab import userdata
    model_name_or_path = "CompendiumLabs/bge-large-en-v1.5-gguf"
    model_basename = 'bge-large-en-v1.5-f32.gguf'
    model_path = hf_hub_download(
    repo_id=model_name_or_path,
    filename=model_basename,
)
    model = Llama(model_path, embedding=True)

    # st.success("Loaded NLP model from Hugging Face!")  # 👈 Show a success message
    
    

    model_2_name = "TheBloke/zephyr-7B-beta-GGUF"
    model_2base_name = "zephyr-7b-beta.Q4_K_M.gguf"
    model_path_model = hf_hub_download(
    repo_id=model_2_name,
    filename=model_2base_name,
)   
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
    model_path=model_path_model,
    temperature=0.75,
    max_tokens=2500,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True, 
    n_ctx=2048,
    n_threads = 2# Verbose is required to pass to the callback manager
)

 
    prompt_template = "<|system|>\
</s>\
<|user|>\
{prompt}</s>\
<|assistant|>"
    template = prompt_template
    prompt = PromptTemplate.from_template(template)
    # st.success("prompt loaded")
    return model, llm, prompt

    
st.title("Please ask your question on Lithuanian rules for foreigners.")
model,llm, prompt  = load_model()
apikey = st.secrets["apikey"] 
pc = Pinecone(api_key=apikey)
index = pc.Index("law")
question = st.text_input("Enter your question:")

if question != "":
    query = model.create_embedding(question)
    # st.write(query)
    q = query['data'][0]['embedding']
    
    response = index.query(
    vector=q,
    top_k=1,
    include_metadata = True,
    namespace = "ns1"
    )
    response_t = response['matches'][0]['metadata']['text']
    # st.write(response_t)
    response = prompt.format(prompt =f"Based on this {response_t} , answer this {question}.")
    st.write_stream(llm.stream(response))
    st.write(response_t)

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
    cache_dir= '/content/models' # Directory for the model
)
    model = Llama(model_path, embedding=True)

    st.success("Loaded NLP model from Hugging Face!")  # 👈 Show a success message
    
    
    # pc = Pinecone(api_key=api_key)
    # index = pc.Index("law")
    model_2_name = "TheBloke/zephyr-7B-beta-GGUF"
    model_2base_name = "zephyr-7b-beta.Q4_K_M.gguf"
    model_path_model = hf_hub_download(
    repo_id=model_2_name,
    filename=model_2base_name,
    cache_dir= '/content/models' # Directory for the model
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
    st.success("loaded the second NLP model from Hugging Face!") 
#     prompt_template = "<|system|>\
# </s>\
# <|user|>\
# {prompt}</s>\
# <|assistant|>"
#     template = prompt_template
#     prompt = PromptTemplate.from_template(template)
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#     llm = LlamaCpp(
#     model_path=model_path_model,
#     temperature=0.75,
#     max_tokens=2500,
#     top_p=1,
#     callback_manager=callback_manager,
#     verbose=True, 
#     n_ctx=2048,
#     n_threads = 2# Verbose is required to pass to the callback manager
# )
    return model, llm
    
st.title("Please ask your question on Lithuanian rules for foreigners.")
model,llm  = load_model()
pc = Pinecone(api_key="003117b0-6caf-4de4-adf9-cc49da6587e6")
index = pc.Index("law")
question = st.text_input("Enter your question:")
query = model.create_embedding(question)
q = query['data'][0]['embedding']
response = index.query(
  vector=q,
  top_k=1,
  include_metadata = True,
  namespace = "ns1"
)
response_t = response['matches'][0]['metadata']['text']
st.header("Answer:")
st.write(response_t)

# if question:
#     # Perform Question Answering
#     answer = qa_chain(context=context, question=question)
    
#     # Display the answer
#     st.header("Answer:")
#     st.write(answer)

Question Answer streaming using simple RAG based on Republic of Lithuania Legal Status of Foreigners law, to ask and answer.

Uses 4v Core CPU (ONLY) on Github codespaces.

Has [Llama-Cpp-Python](https://github.com/abetlen/llama-cpp-python) , [Pinecone](https://app.pinecone.io/), [Streamlit](https://streamlit.io/) 
Using[TheBloke/zephyr-7B-beta-GGUF](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF) as a LLM 
And [CompendiumLabs/bge-large-en-v1.5-gguf](https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf) as an embedding model, k =1 nearest neighbour. 
At the end, it displays the most similar Article found ( to the prompt).


https://github.com/helloworld53/rag_lt/assets/97686596/adc43427-c3dd-4883-b3e8-6b3ed8009e3b



---
title: Rag Lithuania
emoji: 💻
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.32.2
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

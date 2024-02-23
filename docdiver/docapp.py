import streamlit as st
from sentify.main import sentify, known_types
from deepllm.api import *
from main import SourceDoc

with st.sidebar:
    doc_type = st.radio('Document type?', known_types, index=0, horizontal=True)

    doc_name = st.text_input('Link to document or file name?', value="")

    saved_file_name = st.text_input('To be locally saved as?', value="")

    sum_len = st.slider('Sentences in summary?', min_value=2, max_value=20, value=8)

    choice = st.sidebar.radio('LLM?', ['GPT-4', 'GPT-3.5', 'Local LLM'], horizontal=True)



    local = choice == 'Local LLM'

    if local:

        LOCAL_PARAMS['API_BASE'] = st.sidebar.text_input('Local LLM server:', value=LOCAL_PARAMS['API_BASE'])
        local_model()
    else:

        if choice == 'GPT-4':
            smarter_model()
        else:
            cheaper_model()
        key = os.getenv("OPENAI_API_KEY")
        if not key or key == 'EMPTY':
            key = st.text_input("Enter your OPENAI_API_KEY:", "", type="password")
            if not key:
                st.write('Please enter your OPENAI_API_KEY!')
                exit(0)
            else:
                set_openai_api_key(key)


def as_local_file_name():
    if not saved_file_name:
        if doc_type == 'url':
            file_name = doc_name.split('/')[-1]
        else:
            file_name = doc_name
    else:
        file_name = saved_file_name

    file_name = file_name.replace('.pdf', '.txt').replace('.PDF', '.txt')
    return file_name


def process_it():
    sd = SourceDoc(doc_type=doc_type, doc_name=doc_name, threshold=0.5, top_k=3)

    summary=sd.summarize(best_k=sum_len)

    st.write(summary)


st.sidebar.button('Process it!', on_click=process_it)

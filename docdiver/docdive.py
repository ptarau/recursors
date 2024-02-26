import streamlit as st
from deepllm.api import *
from main import SourceDoc


def handle_uploaded():
    if st.session_state.uploaded_file is None:
        return None
    fpath = save_uploaded_file()
    suf = fpath[-4:].lower()
    # fname = fpath[:-4]
    if suf == ".pdf" or suf == ".txt":
        return fpath
    else:
        st.write("UPLOAD .txt / .pdf file!")


def save_uploaded_file():
    upload_dir = './docs/'
    fname = st.session_state.uploaded_file.name
    fpath = os.path.join(upload_dir, fname)
    if exists_file(fpath):
        return fpath
    ensure_path(upload_dir)
    with open(fpath, "wb") as f:
        f.write(st.session_state.uploaded_file.getbuffer())
    return fpath


with st.sidebar:
    doc_type = st.radio('Document type?', ('url', 'local pdf or txt file'), index=0, horizontal=True)

    if doc_type == 'local pdf or txt file':
        st.session_state.uploaded_file = st.file_uploader(
            "SELECT FILE", type=["txt", "pdf"]
        )
        doc_name = handle_uploaded()
        if doc_name is None:
            pass
        elif doc_name.lower().endswith('.pdf'):
            doc_type = 'pdf'
        elif doc_name.lower().endswith('.txt'):
            doc_type = 'txt'
        else:
            st.write('Unable to process:', doc_name)

    else:
        doc_name = st.text_input('Link to document name?', value="")

    sent_count = st.slider('Number of salient sentences to work with?', min_value=3, max_value=300, value=30)

    choice = st.sidebar.radio('LLM?', ['GPT-4', 'GPT-3.5', 'Local LLM'], horizontal=True)

    if choice == 'GPT-4':
        smarter_model()
    elif choice == 'GPT-3.5':
        cheaper_model()
        sent_count = min(80, sent_count)
    else:
        LOCAL_PARAMS['API_BASE'] = st.sidebar.text_input(
            'Local LLM server:', value=LOCAL_PARAMS['API_BASE']
        )
        local_model()
        sent_count = min(80, sent_count)

    processing = st.sidebar.radio('What to do?', ['Summary', 'Review', 'Salient sentences'], horizontal=True)

key = os.getenv("OPENAI_API_KEY")
if not key or key == 'EMPTY':
    key = st.text_input("Enter your OPENAI_API_KEY:", "", type="password")
    if not key:
        st.write('Please enter your OPENAI_API_KEY!')
        exit(0)
    else:
        set_openai_api_key(key)


def process_it():
    if not doc_name:
        if doc_type == 'url':
            mes = "Please enter a URL!"
        else:
            mes = "Please upload a file!"
        st.write(mes)
        return

    sd = SourceDoc(doc_type=doc_type, doc_name=doc_name, threshold=0.5, top_k=3)

    if processing == 'Summary':
        result = sd.summarize(best_k=sent_count)
    elif processing == 'Salient sentences':
        result = dict(
            (i, s.strip()) for (i, s) in sd.extract_summary(best_k=sent_count)
        )
    else:
        assert processing == 'Review', processing
        result = sd.review(best_k=sent_count)

    st.write(result)


st.sidebar.button('Process it!', on_click=process_it)

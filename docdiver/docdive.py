import streamlit as st
from deepllm.api import *
from main import SourceDoc

if 'history' in st.session_state:
    history = st.session_state.history
else:
    history = dict()
    st.session_state.history = history


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
    st.write('**DocDiver is a DeepLLM application**')
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

    sent_count = st.slider('Number of salient sentences to work with?',
                           min_value=3, max_value=300, value=30)

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

    processing = st.sidebar.radio('What to do?',
                                  ['Summary',
                                   'Extract salient sentences',
                                   'Review',
                                   'Talk about the document',
                                   'Show history',
                                   'Clear history'
                                   ],
                                  horizontal=True
                                  )

    quest = st.text_area(label='Question', value="", key='quest')

key = os.getenv("OPENAI_API_KEY")
if not key or key == 'EMPTY':
    key = st.text_input("Enter your OPENAI_API_KEY:", "", type="password")
    if not key:
        st.write('Please enter your OPENAI_API_KEY!')
        exit(0)
    else:
        set_openai_api_key(key)


def process_it():
    global history
    if not doc_name:
        if doc_type == 'url':
            mes = "Please enter a URL!"
        else:
            mes = "Please upload a file!"
        st.write(mes)
        return

    sd = SourceDoc(doc_type=doc_type, doc_name=doc_name, threshold=0.5, top_k=3)
    result=""

    if processing == 'Show history':
        for k, v in history.items():
            st.write(k)
            st.write(v)
            st.write()


    elif processing == 'Clear history':
        history = dict()
        st.session_state.history = history


    elif processing == 'Summary':
        result = sd.summarize(best_k=sent_count)
        history[processing] = result
    elif processing == 'Extract salient sentences':
        result = dict(
            (i, s.strip()) for (i, s) in sd.extract_summary(best_k=sent_count)
        )
        history[processing] = result
    elif processing == 'Review':
        result = sd.review(best_k=sent_count)
        history[processing] = result
    else:
        assert processing == 'Talk about the document', processing
        st.write('Question:', quest)
        result, follow_up = sd.ask(quest)
        st.session_state.quest = follow_up

        talk = history.get(processing, [])
        talk.append('Q: ' + quest)
        talk.append('A: ' + result)
        history[processing] = talk

    st.session_state.history = history
    st.write(result)
    st.write(f'COSTS: ${round(sd.dollar_cost(), 4)},  TIME: {sd.time} seconds')


st.sidebar.button('Proceed!', on_click=process_it)

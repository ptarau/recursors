import streamlit as st
from main import *
from deepllm.tools import file2string

UPLOAD_DIR = './UPLOAD_DIR/'

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
    upload_dir = UPLOAD_DIR
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
    doc_type = st.radio('Document type?', ('local pdf or txt file', 'url'), index=0, horizontal=True)

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
                           min_value=3, max_value=300, value=100)

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

    center = st.text_input('Center of attention?', value="", help="Results with be centered on these key concepts")

    processing = st.sidebar.radio('What to do?',
                                  ['Summary',
                                   'Review',
                                   'Salient sentences',
                                   'Relation graph',
                                   'Chat about it',
                                   'History'
                                   ],
                                  horizontal=True
                                  )

    quest = st.text_area(label='Question', value="", key='quest')


def clear_key():
    API_KEY[0] = ""


def collect_key():
    key = os.getenv("OPENAI_API_KEY")
    if key and len(key) > 40:
        set_openai_api_key(key)
    key = ensure_openai_api_key()
    if not key and not IS_LOCAL_LLM[0]:
        key = st.text_input("Enter your OPENAI_API_KEY:", "", type="password")
        if not key:
            st.write('And do not forget to clear it when done with the app!')
            exit(0)
        else:
            set_openai_api_key(key)


collect_key()


def clear_it():
    global history
    if clearing == 'History':
        history = dict()
        st.session_state.history = history
        st.write('Cleared history')
    elif clearing == 'Caches':
        extradirs = [SENT_CACHE, SENT_STORE_CACHE, UPLOAD_DIR, 'lib']
        dirs = clear_caches() + extradirs
        for x in extradirs:
            remove_dir(x)
        st.cache_data.clear()
        st.write('REMOVED:', dirs)
    elif clearing == 'Openai key':
        clear_key()
        st.write('Cleared key')

def refresh_graph(*args):
    for f in args:
        remove_file(f)

def process_it():
    """
    DO:
           ['Summary',
            'Review',
            'Salient sentences',
            'Relation graph',
            'Chat about it',
            'History'
    """
    t1 = time()
    global history
    if not doc_name:
        if doc_type == 'url':
            mes = "Please enter a URL!"
        else:
            mes = "Please upload a file!"
        st.write(mes)
        return

    if processing == 'History':
        for k, v in history.items():
            st.write(k)
            st.write(v)
            st.write()
        return

    sd = SourceDoc(
        doc_type=doc_type,
        doc_name=doc_name,
        threshold=0.5,
        top_k=3
    )
    result = ""

    if processing == 'Summary':
        result = sd.summarize(best_k=sent_count, center=center)
        history[processing] = result
    elif processing == 'Salient sentences':
        result = dict(
            (i, s.strip()) for (i, s) in sd.extract_summary(best_k=sent_count)
        )
        history[processing] = result
    elif processing == 'Review':
        result = sd.review(best_k=sent_count, center=center)
        history[processing] = result
    elif processing == 'Relation graph':
        hfile, pfile, jfile = sd.show_relation_graph(min(50, sent_count), center=center)
        if hfile is None:
            st.write('Relation graph generation failed for:', doc_name)
            return
        html_code = open(hfile, 'r', encoding='utf-8').read()
        with st.expander('Visualize relation graph'):
            st.components.v1.html(html_code, height=1000, scrolling=True)
        pl_code = file2string(pfile)
        js_code = from_json(jfile)
        with st.expander('Relations as Prolog code'):
            for clause in pl_code.split('\n'):
                st.write(clause)
        with st.expander('Relations as Json code'):
            st.write(js_code)
        st.button('Refresh graph!', on_click=refresh_graph,args=(hfile, pfile, jfile))

    else:
        assert processing == 'Chat about it', processing
        st.write('Question:', quest)
        result, follow_up = sd.ask(quest)
        st.session_state.quest = follow_up

        talk = history.get(processing, [])
        talk.append('Q: ' + quest)
        talk.append('A: ' + result)
        history[processing] = talk

    st.session_state.history = history
    st.write(result)
    st.write(f'COSTS: ${round(sd.dollar_cost(), 4)}')
    st.write('TIMES:', sd.get_times())
    total_time = sum(sd.get_times().values())
    st.write('TOTAL API TIME:', round(total_time, 2), 'seconds')
    t2 = time()
    st.write('TOTAL APP TIME:', round(t2 - t1, 2), 'seconds')


st.sidebar.button('Proceed!', on_click=process_it)
clearing = st.sidebar.radio('What to clear?',
                            [
                                'History',
                                'Caches',
                                'Openai key'
                            ],
                            horizontal=True
                            )
st.sidebar.button('Clear!', on_click=clear_it)

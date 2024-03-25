import streamlit as st

from deepllm.api import *

print('Running DeepLLM as a streamlit app!')

st.set_page_config(layout="wide")

st.title('Streamlit-based [DeepLLM](https://github.com/ptarau/recursors) Demo Client')

prompters = prompter_dict()

def clear_key():
    API_KEY[0]=""

def collect_key():
    key = os.getenv("OPENAI_API_KEY")
    if key and len(key)>40:
        set_openai_api_key(key)
    key=ensure_openai_api_key()
    if not key and not IS_LOCAL_LLM[0]:
        key = st.text_input("Enter your OPENAI_API_KEY:", "", type="password")
        if not key:
            st.write('And do not forget to clear it when done with the app!')
            exit(0)
        else:
            set_openai_api_key(key)

with st.sidebar:
    local = st.sidebar.checkbox('Local LLM?', value=False)

    if local:

        LOCAL_PARAMS['API_BASE'] = st.sidebar.text_input('Local LLM server:', value=LOCAL_PARAMS['API_BASE'])
        local_model()
    else:
        choice = st.sidebar.radio('OpenAI LLM', ['GPT-4', 'GPT-3.5'])
        if choice == 'GPT-4':
            smarter_model()
        else:
            cheaper_model()
        collect_key()

    recursor = st.select_slider('LLM Agent', options=('Advisor', 'Recursor', 'Rater'), value='Recursor')

    threshold = st.slider('Threshold:', 0, 100, 50) / 100

    lim = st.slider('Maximum depth', 1, 4, 1)

    svos = st.toggle('Extract relations?', value=False)

    trace = st.toggle('Show trace?', value=False)

    initiator = st.text_area('Topic to explore:', value='artificial general intelligence')

    prompter_name = st.radio('Prompter:', list(prompters.keys()))

    prompter = prompters[prompter_name]

    query_it = st.button('Activate LLM!')

    show_it = st.button('Visualize relation graph!')

    browse_it = st.button('Browse relation graph in new tab!')


def visualize(data, new_tab=False):
    fname = 'rel_graph'
    url, hfile = vis_svos(data, fname=fname, show=False)
    if new_tab:
        browse(url)
    else:
        html_code = open(hfile, 'r', encoding='utf-8').read()
        st.components.v1.html(html_code, height=1024, scrolling=True)


def do_query():

    if svos:
        activate_svos()
    else:
        deactivate_svos()

    if recursor == 'Recursor':
        g = run_recursor(initiator=initiator, prompter=prompter, lim=lim)
    elif recursor == 'Advisor':
        g = run_advisor(initiator=initiator, prompter=prompter, lim=lim)
    else:
        assert recursor == 'Rater'
        g = run_rater(
            initiator=initiator, prompter=prompter, lim=lim, threshold=threshold)

    st.write('STARTING!')
    if trace:  st.write('TRACE')
    for kind, data in g:
        if kind == 'TRACE':
            if trace: st.write(data)
        elif kind == 'PROMPTER':
            if trace: st.write(data)
        else:
            st.write(kind)
            if kind == 'CLAUSES':
                st.code(show_clauses(data), language='prolog')
            elif kind == 'MODEL':
                st.code(show_model(data))
            elif kind == 'SVOS':
                st.code(show_svos(data))
                return data
            else:
                assert kind == 'COSTS'
                st.write(data)


if 'svo_data' not in st.session_state:
    st.session_state.svo_data = None

if query_it:
    st.session_state.svo_data = do_query()

if st.session_state.svo_data is not None and (show_it or browse_it):
    new_tab = False
    if browse_it:
        new_tab = True
    visualize(st.session_state.svo_data, new_tab=new_tab)

if not IS_LOCAL_LLM[0]:
    st.sidebar.button('Clear OpenAI key!', on_click=clear_key)

import streamlit as st

from inquisitor import *

print('Running DeepQA as a streamlit app!')

st.set_page_config(layout="wide")

st.sidebar.title('[DeepQA](https://github.com/ptarau/recursors/tree/main/deepQA): a DeepLLM App exploring self-generated follow-up questions, version  '+get_version())

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
        choice = st.sidebar.radio('OpenAI LLM', ['GPT-4o', 'GPT-4o-mini'])
        if choice == 'GPT-4o':
            smarter_model()
        else:
            cheaper_model()

        collect_key()

    lim = st.slider('Maximum depth', 1, 4, 1)

    show_dcg = st.toggle('Show generated Definite Clause Grammar?', value=False)

    # initiator = st.text_area("Question to start with:', value='How does a vector db work in hybrid mode with keyword search?")

    initiator = st.text_area('Question to start with:',
                             value="How can humans align superintelligent LLMs to human goals and values?")

    query_it = st.button('Activate LLM!')


def do_query():

    def printer(*xs):
        st.write(*xs)

    qe = QuestExplorer(
        initiator=initiator,
        prompter=quest_prompter,
        lim=lim,
        local=local)
    qe.run(printer=printer)

    if show_dcg:
        dcg = qe.show_dcg()
        assert dcg is not None
        st.write(":blue[DCG GRAMMAR:]")
        lines = dcg.split("\n")
        for line in lines:
            st.write(":green[" + line + "]")


if query_it:
    do_query()

if not IS_LOCAL_LLM[0]:
    st.sidebar.button('Clear OpenAI key!', on_click=clear_key)

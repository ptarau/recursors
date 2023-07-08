import openai

import streamlit as st

from deepllm.api import *

print('Running DeepLLM as a streamlit app!')

st.set_page_config(layout="wide")

st.title('Streamlit-based DeepLLM Client')

key = ensure_openai_api_key(
    st.sidebar.text_area("Unless it is in your environment, enter your OPENAI_API_KEY:", ""))

d = prompter_dict()
# st.write(d)


with st.sidebar:
    recursor = st.radio('LLM Agent:', ['Recursor', 'Advisor', 'Rater'])  # , 'Truth_rater'])

    threshold = st.slider('Threshold:', 0, 100, 50)/100

    lim = st.slider('Maximum depth', 1, 4, 1)

    initiator = st.text_area('Topic to explore:', value='Origin of COVID-19')

    prompter_name = st.radio('Prompter:', list(d.keys()))

    prompter = eval(d[prompter_name])

    query_it = st.button('Activate LLM!')


def do_query():
    if recursor == 'Recursor':
        g = run_recursor(initiator=initiator, prompter=prompter, lim=lim)
    elif recursor == 'Advisor':
        g = run_advisor(initiator=initiator, prompter=prompter, lim=lim)
    else:
        assert recursor == 'Rater'
        g = run_rater(
            initiator=initiator, prompter=prompter, lim=lim, threshold=threshold)

    st.write('STARTING:')
    for x in g:
        # st.write('.')
        st.write(*x)
    st.write('DONE:')


if query_it:
    do_query()

import streamlit as st

from deepllm.api import *

print('Running DeepLLM as a streamlit app!')

st.set_page_config(layout="wide")

st.title('Streamlit-based [DeepLLM](https://github.com/ptarau/recursors) Demo Client')

prompters = prompter_dict()

key = None
# st.write('Key:',key)

with st.sidebar:
    # recursor = st.radio('LLM Agent:', ['Recursor', 'Advisor', 'Rater'])  # , 'Truth_rater'])

    recursor = st.select_slider('LLM Agent', options=('Advisor', 'Recursor', 'Rater'), value='Recursor')

    threshold = st.slider('Threshold:', 0, 100, 50) / 100

    lim = st.slider('Maximum depth', 1, 4, 1)

    trace = 'on' == st.select_slider('Trace', options=('off', 'on'), value='off')

    smarter = 'Smarter: gpt-4' == st.select_slider(
        'LLM model', options=('Cheaper: gpt-3.5-turbo', 'Smarter: gpt-4'),
        value='Smarter: gpt-4'
    )

    key = st.sidebar.text_input("Enter your OPENAI_API_KEY:", "", type="password")

    initiator = st.text_area('Topic to explore:', value='Superhuman artificial general intelligence')

    prompter_name = st.radio('Prompter:', list(prompters.keys()))

    prompter = prompters[prompter_name]

    query_it = st.button('Activate LLM!')


def do_query():
    assert key
    assert len(key) > 40
    set_openai_api_key(key)

    if smarter:
        smarter_model()
    else:
        cheaper_model()

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
            else:
                assert kind == 'COSTS'
                st.write(data)

    st.write('DONE!')


if query_it:
    do_query()

import streamlit as st

from deepllm.api import *
from deepllm.interactors import Agent

from deepllm.questmaker import make_agent, one_quest

st.set_page_config(layout="wide")

st.sidebar.title(
    ":blue[[AutoQA](https://github.com/ptarau/recursors/tree/main/autoQA): DeepLLM app with Follow-up Question Generator]")

local = st.sidebar.checkbox('Local LLM?', value=False)

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
            st.write('Please enter your OPENAI_API_KEY!')
            exit(0)
        else:
            set_openai_api_key(key)

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


def clean_quest(text):
    return ' '.join(text.split())


chat_name = st.sidebar.text_input('Chat name?', 'autoQA')

question = clean_quest(st.sidebar.text_area("ENTER QUESTION:", key='quest'))

agent=None

def do_answers():
    global agent
    if agent is None:
        agent = make_agent(name=chat_name)

    st.write('\nQ:', question)

    context = agent.initiator
    if context is None: context = "think like a genius"
    answer, new_question = one_quest(agent, question, context)
    assert agent.initiator

    st.write('\nA:', answer)

    st.session_state.quest = new_question

    mem_stats(agent)
    if agent.initiator is not None:
        st.write('INITIATOR:')
        st.write([agent.initiator])
    show_mem('SHORT_TERM MEMORY:', agent.short_mem)
    show_mem('LONG_TERM MEMORY:', agent.long_mem)


def clear_cache():
    global agent
    if agent is None:
        agent = make_agent(name=chat_name)
    st.write('CLEARING CACHE AT:', agent.cache_name())
    agent.clear()
    st.cache_resource.clear()


def mem_stats(agent):
    st.write('SHORT_TERM_MEMORY SIZE:',
             len(agent.short_mem),
             'LONG_TERM_MEMORY SIZE:',
             len(agent.long_mem), 'COSTS:', round(agent.dollar_cost(), 4))


def show_mem(name, mem):
    st.write('*' + name + '*')
    qas = []
    for x in mem.values():
        qas.append(x)
    st.write(qas)


st.sidebar.button("COMPUTE ANSWER", on_click=do_answers)

st.sidebar.button('CLEAR CACHES', on_click=clear_cache)

if not IS_LOCAL_LLM[0]:
    st.sidebar.button('Clear OpenAI key!', on_click=clear_key)


# st.sidebar.button('SHOW HISTORY', on_click=show_mem)

def examples():
    return """
How can Logic Programming enhance Generative AI systems?

How to teach Logic Programming as a first graduate course?

How to best teach Prolog in an undergrad course?

What are the advantages of not using quantifiers explictely  in logic programming languages like Prolog?

What are the advantages of not using quantifiers explicitly in logic programming languages like Prolog?

Can negation as failure be avoided in propositional Horn Clause logic by using 
integrity constraints (sets of facts that can not be all true at the same time)
to benefit from the linear complexity of its fixpoint computation?

Would you teach an "Introduction to Prolog" course or an "Introduction to Logic" course
to help students develop computational thinking?
"""

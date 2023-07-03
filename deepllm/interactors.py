import time
import openai
import tiktoken
from string import Template
from .params import *
from .tools import *

openai.api_key = os.getenv("OPENAI_API_KEY")


# tools

def count_toks(text):
    enc = tiktoken.get_encoding("gpt2")
    toks = enc.encode(text)
    return len(toks)


def dict_trim(d):
    k = next(iter(d))
    v = d.pop(k)
    return k, v


# basic building blocks

def clean_pattern(p):
    ps = p.split('\n')
    ps = [p.strip() for p in ps]
    return ' '.join(ps)  # +"\n\n"


class Agent:
    """
    manages all aspects of the interaction with LLMs

    """

    def __init__(self, name):
        self.prompter()
        self.tracker()
        self.cacher()
        self.tuner()
        self.talker()
        self.name = name
        self.CACHES=None
        self.TRACE=None
        PARAMS()(self)
        # print('AGENT !!!!',self.__dict__)

    def tuner(self,
              model="gpt-3.5-turbo",
              temperature=0.2,
              n=1,
              max_toks=4000):
        """
          GPT parameter tuners
        """
        self.model = model
        self.temperature = temperature
        self.n = n
        self.max_toks = max_toks

    def cacher(self):
        """
           caching mechanisms for
           interaction state, including all inherited attributed

           we are saving things to readable .json files but
           overriders might pickle, compress or persist to a database
        """
        pass

    def cache(self):
        return self.CACHES + self.name + ".json"

    def persist(self):
        """
        collects and persits all appropriate attributs
        """
        if self.name is None: return

        if self.TRACE: print('PERSISTING:', self.cache())

        kvs = []
        d = self.__dict__
        for k, v in d.items():
            if any(map(lambda t: isinstance(v, t), [int, float, str, list, tuple, dict])):
                kvs.append((k, v))

        ensure_path(self.cache())
        to_json(kvs, self.cache())

    def resume(self):
        """
        resumes all attributes from their
        persisted values
        notes that these might override current attributes
        coming from wherever they might be inherited from

        if needed, use clear to clear all stored states
        """
        if self.name is None: return

        if not exists_file(self.cache()): return
        kvs = from_json(self.cache())
        for k, v in kvs:
            setattr(self, k, v)

    def clear(self):
        if self.name is None: return
        if exists_file(self.cache()):
            remove_file(self.cache())

    def tracker(self):
        """
           manages the API's parameters, collects an cleans-up answers
           remembers past interactions in short and long-term emmory and
           avoids calling the API twice on the same quary that it retrives
           from its memory
           """
        self.short_mem = dict()
        self.long_mem = dict()
        self.prompt_toks = 0
        self.compl_toks = 0
        self.sys_prompt = "I am a helpful assistant."

    def to_message(self, quest):
        """
        uses its past interaction in short-term memory
        as context to build the message to be sent to the API
        """
        mes = [dict(role='system', content=self.sys_prompt)]
        for (q, a) in self.short_mem.items():
            assert isinstance(q, str), q
            qd = dict(role='user', content=q)
            ad = dict(role='assistant', content=a)
            mes.extend([qd, ad])
        mes.append(dict(role='user', content=quest))
        return mes

    def already_answered(self, quest):
        """
        retrieves already answred questions from its memory
        """
        answer = self.short_mem.get(quest, None)
        if answer is not None: return answer
        answer = self.long_mem.get(quest, None)
        return answer

    def trim_context(self, quest, max_toks):
        """
        moves oldest items from short-term to long-term memory
        to avoid token-count overflows
        """

        p_toks = 3 * count_toks(quest)

        toks = []
        for k, v in self.short_mem.items():
            toks.append(count_toks(k) + count_toks(v))

        for i in range(len(toks)):
            tok_estimate = sum(toks[i:]) + p_toks
            if tok_estimate < max_toks: break

            k, v = dict_trim(self.short_mem)
            self.long_mem[k] = v

    def spill(self):
        """
        programmatically spills the content of its short term
        memory ot its long-term memory - a way to forget current context
        """
        for k, v in self.short_mem.items():
            self.long_mem[k] = v
        self.short_mem = dict()
        return self

    def prompter(self):
        """
           prompt generators:
           a prompt can be either the actual question
           or the result of substituting in a template values from a dict
        """
        pass

    def set_pattern(self, pattern):
        """
        sets the prompter's pattern to be filled later
        with specific data to generate actual prompts
        """
        self.pattern = clean_pattern(pattern)

    def apply_prompt(self, quest):
        """
        fills out the Template associated to the pattern
        with values in a dictionary, instantiating the
        $-marked variables  corresponding to keys in the dict
        """
        if isinstance(quest, str):
            assert self.pattern is None
            return quest
        assert self.pattern is not None
        template = Template(self.pattern)
        return template.substitute(dict(quest))

    def talker(self):
        """
           talking to the LLM:
           assumes all other components already inherited
           by the DefaultInteractor
           """
        pass

    def ask(self, *args, **kwargs):
        """
        asks the LLM via the API and returns a
        list of n answers (usually just n=1)
        while memoizing the prompt-answer pair and
        computing the token costs
        """
        h = tuple(kwargs.items())
        if not h:
            assert len(args) == 1, ('BAD args', args)
            quest0 = args[0]
            assert isinstance(quest0, str)
        else:
            quest0 = h

        quest = self.apply_prompt(quest0)

        answered = self.already_answered(quest)
        if answered is not None:
            return answered

        self.trim_context(quest, self.max_toks)
        mes = self.to_message(quest)

        max_attempts = 3
        r, t = None, None
        pt, ct = None, None

        for attempt in range(max_attempts):
            try:
                r = openai.ChatCompletion.create(
                    model=self.model,
                    messages=mes,
                    temperature=self.temperature,
                    n=self.n
                )
                pt = r['usage']['prompt_tokens']
                ct = r['usage']['completion_tokens']

                break
            except Exception as e:
                if attempt >= max_attempts - 1:
                    print('\n\n ***GPT exception:', e)
                    raise e
                else:
                    print('retrying: ', attempt)
                    time.sleep(0.5)

        def res(i):
            assert isinstance(i, int), i
            return r['choices'][i]['message']['content'].strip()

        answers = [res(i) for i in range(self.n)]
        if self.n > 1:
            answer = "\n".join([spacer(a) for a in answers])  # one answer per line
        else:
            answer = answers[0]
        assert isinstance(answer, str), answer

        self.prompt_toks += pt
        self.compl_toks += ct

        answer = self.post_process(quest0, answer)

        self.short_mem[quest] = answer
        assert isinstance(answer, str), answer
        return answer

    def post_process(self, _quest0, answer):
        """
        to be overriden if application-specific post-processing
        of the LLM's answer is required
        """
        return answer

    def dollar_cost(self):
        """
        computes API costs for several models
        to be extended as new models appear
        """
        if self.model == 'gpt-3.5-turbo':
            return (self.prompt_toks * 0.0015 + self.compl_toks * 0.002) / 1000
        if self.model == 'gpt-3.5-turbo-16k':
            return (self.prompt_toks * 0.003 + self.compl_toks * 0.004) / 1000
        if self.model == 'gpt-4':
            return (self.prompt_toks * 0.03 + self.compl_toks * 0.06) / 1000
        if self.model == 'gpt-4-32k':
            return (self.prompt_toks * 0.06 + self.compl_toks * 0.12) / 1000
        return 0.0 # case of local LLM

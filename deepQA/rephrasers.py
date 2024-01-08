# In the style of Wittgenstein's Tractatus describe all the facts that describe an Abrams tank.

from sentify.main import sentify, text2file
from deepllm.interactors import Agent, PARAMS
from deepllm.recursors import SvoMaker

witt_prompter1_txt = dict(
    name="tractatus_style",
    target=".txt",
    prompt="""
 In the spirit and notation style of Wittgenstein's Tractatus
 render the tree of atomic facts that summarize the following text:
 "$text".
 """
)

witt_prompter_txt = dict(
    name="tractatus_style",
    target=".txt",
    prompt="""
 In the style of Wittgenstein's Tractatus summarize the essential
 atomic facts that he would state about the following text. 
 Return the result as tree representing details following 
 the numerotation scheme of the Tractatus.
 Here is the text to work on: 
 
 "$text".
    """
)

svo_prompter = dict(
    name="fact_to_svo",
    target='.txt',
    rel_p="""Split the following sentence into an SVO triplet.
    Trim down the subject and object parts to their essential noun phrases.
    Here is the sentence:
       
    "$text$
    """
)


class Factualizer(Agent):
    # def __init__(self, name, **kwargs):
    #    super().__init__(name, **kwargs)

    def factify(self, prompter, kind, source):
        CF = PARAMS()
        self.set_pattern(prompter['prompt'])
        self.outf = CF.OUT + self.name + "_" + prompter['name'] + prompter['target']
        sents = sentify(kind, source)
        text = "\n".join(sents)
        text = self.ask(text=text)  # text is the

        return text

    def post_process(self, _quest, text):
        lines = text.split('\n')
        keepers = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not line[0].isdigit():
                continue
            keepers.append(line)
        return "\n".join(keepers)


class Relationizer(Factualizer):
    def relationize(self, prompter, kind, source):
        text = self.factify(prompter, kind, source)
        lines = text.split("\n")
        pairs = []
        for line in lines:
            n, _, t = str.partition(line, " ")
            pairs.append((n, t))
        return pairs


def test_relationizer():
    page = 'horn_clause'
    agent = Relationizer(page)
    text = agent.relationize(witt_prompter_txt, 'wikipage', page)
    print(text)
    #text2file(text, agent.outf)
    print(agent.dollar_cost())


def test_rephraser():
    page = 'logic_programming'
    agent = Factualizer(page)
    text = agent.factify(witt_prompter_txt, 'wikipage', page)
    print(text)
    text2file(text, agent.outf)
    print('COST:', agent.dollar_cost())


def test_rephraser2():
    # url = 'https://arxiv.org/pdf/1904.11694.pdf'
    url = 'https://arxiv.org/pdf/1912.10824.pdf'
    agent = Factualizer('url_dif')
    text = agent.factify(witt_prompter_txt, 'url', url)
    print(text)
    text2file(text, agent.outf)
    print(agent.dollar_cost())


def test_rephraser3():
    txt = './data/gpl.txt'
    agent = Factualizer('txt_gpl')
    text = agent.factify(witt_prompter_txt, 'txt', txt)
    print(text)
    text2file(text, agent.outf)
    print(agent.dollar_cost())


if __name__ == "__main__":
    #test_rephraser()
    test_relationizer()

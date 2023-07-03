from .embedders import Embedder
from .prompters import *
from .recursors import *


class Advisor(AndOrExplorer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pname = decision_prompter['name']
        oname = f'{self.name}_{pname}'
        self.oracle = Agent(name=oname)
        self.oracle.set_pattern(decision_prompter['decider_p'])

    def appraise(self, g, _trace):
        # xs = to_list((g, trace))
        # context = ", ".join(xs)

        advice = just_ask(self.oracle, g=g, context=self.initiator)

        print('!!! ADVICE for:', g, advice)

        return advice.startswith('True')

    def resume(self):
        super().resume()
        self.oracle.resume()

    def persist(self):
        super().persist()
        self.oracle.persist()

    def costs(self):
        d = super().costs()
        d['oracle'] = self.oracle.dollar_cost()
        return d


class Rater(AndOrExplorer):
    def __init__(self, threshold=None, **kwargs):
        super().__init__(**kwargs)
        pname = ratings_prompter['name']
        oname = f'{self.name}_{pname}'
        self.oracle = Agent(oname)
        self.oracle.set_pattern(ratings_prompter['rater_p'])
        self.threshold = threshold

    def appraise(self, g, _trace):

        advice = ask_for_clean(self.oracle, g=g, context=self.initiator)

        if not advice:
            print('*** NO ADVICE FOR:', g)
            return False

        rating = advice[0].strip()

        print(f'\n-----EXPLANATION {rating}\n---\n')
        rating = rating.split('|')[0].strip()
        if ' ' in rating: rating = rating.split()[1]
        try:
            f = float(rating)
        except Exception:
            print('*** UNPARSED RATING:', rating)
            f = 5
        f = f / 100.0

        ok = f >= self.threshold

        print(f'RATING of "{g}" w.r.t "{self.initiator}" is {round(f, 4)} --> {ok}')

        return ok

    def resume(self):
        super().resume()
        self.oracle.resume()

    def persist(self):
        super().persist()
        self.oracle.persist()

    def costs(self):
        d = super().costs()
        d['oracle'] = self.oracle.dollar_cost()
        return d


def load_ground_truth(truth_file='logic_programming'):
    with open(f'{PARAMS().DATA}{truth_file}.txt', 'r') as f:
        sents = f.read().split('\n')
    return [s for s in sents if s]


class TruthRater(AndOrExplorer):
    """
    recursor enhanced with ability to look-up
    how close a given fact is to the set of
    gound-truth facts
    """

    def __init__(self, truth_file=None, threshold=None, **kwargs):
        assert None not in (truth_file, threshold)
        super().__init__(**kwargs)
        self.threshold = threshold
        self.store = Embedder(truth_file)
        self.truth_file = truth_file
        if not exists_file(self.store.cache()):
            sents = load_ground_truth(truth_file=truth_file)
            self.store.store(sents)
        self.top_k = PARAMS().TOP_K

    def appraise(self, g, _trace):
        sents_rs = self.store.query(g, self.top_k)
        z = list(zip(*sents_rs))
        r = sum(z[1]) / self.top_k
        # sents=map(str,sents_rs)

        # print('!!!!!', r, '>', self.threshold)
        if r > self.threshold:
            ok = True
        else:
            ok = False
        print(f'RATING of "{self.initiator}->{g}" w.r.t truth in "{self.truth_file}.txt" is {round(r, 4)} --> {ok}')
        print('AS AVG. OF NEAREST SENTS:')
        for sent, r in sents_rs: print(sent, '->', round(r, 4))
        return ok

class AbstractMaker:
    def __init__(self, topic=None, keywords=None):
        assert None not in (topic, keywords)
        self.topic = topic
        self.keywords = keywords
        prompter = sci_abstract_maker
        pname = prompter['name']
        tname = topic.replace(' ', '_').lower()
        self.agent = Agent(f'{tname}_{pname}')
        self.agent.set_pattern(prompter['writer_p'])
        PARAMS()(self)

    def run(self):
        return ask_for_clean(self.agent, g=self.topic, context=self.keywords)

from collections import defaultdict
from deepllm.params import *
from deepllm.interactors import Agent, clean_up, to_list, from_text
from deepllm.horn_prover import qprove
from deepllm.tools import in_stack
from deepllm.prompters import *
from deepllm.vis import visualize_rels


def ask_for_clean(agent, g, context):
    answer = agent.ask(g=g, context=context)
    agent.spill()
    print('ANSWER:',answer)

    xs = from_text(answer)
    res = clean_up(xs)
    return res


def just_ask(agent, g, context):
    answer = agent.ask(g=g, context=context)
    agent.spill()
    return answer


class Unfolder:
    """
    return pair of AND/OR agents
    steering LLM interactions
    """

    def __init__(self, name, prompter, lim):
        pname = prompter['name']
        and_name = f'{name}_{pname}_and_{lim}'
        or_name = f'{name}_{pname}_or_{lim}'

        self.and_ = Agent(and_name)
        self.or_ = Agent(or_name)

        self.and_.set_pattern(prompter['and_p'])
        self.or_.set_pattern(prompter['or_p'])

    def ask_and(self, goal, context):
        return ask_for_clean(self.and_, goal, context)

    def ask_or(self, goal, context):
        return ask_for_clean(self.or_, goal, context)

    def persist(self):
        self.and_.persist()
        self.or_.persist()

    def resume(self):
        self.and_.resume()
        self.or_.resume()

    def costs(self):
        return {'and': self.and_.dollar_cost(), 'or': self.or_.dollar_cost()}


class AndOrExplorer:
    """
    Simple Prolog-inspired recursive descent steering LLMs to
    expand conjunctively or disjunctively while
    staying focussed via a controlled history of
    explored goals while unfolding new ones.
    Besides returning a stream of answers we
    also generate a Prolog program to be further
    explored with logic programming tools.
    """

    def __init__(self, initiator=None, prompter=None, lim=1, strict=False):
        assert initiator is not None
        assert prompter is not None
        self.initiator = " ".join(initiator.replace('.', ' ').strip().split())
        self.name = self.initiator.lower().strip().replace(' ', '_')
        self.prompter = prompter
        self.pname = prompter['name']
        self.lim = lim
        self.strict = strict
        self.unf = Unfolder(self.name, prompter, lim)
        self.clauses = defaultdict(list)
        self.facts = dict()
        self.svo = None
        self.OUT = None
        PARAMS()(self)

    def new_clause(self, g, trace):
        """
        invents a set of new clauses
        given a goal and the trace of
        the past steps leading to g
        """
        or_context = to_context(trace, self.initiator)
        hs = self.unf.ask_or(g, or_context)
        and_context = to_context((g, trace), self.initiator)
        for h in hs:
            if h == g or in_stack(h, trace): continue
            bs = self.unf.ask_and(h, and_context)  # invent their bodies
            if h in bs: continue
            yield h, bs

    def costs(self):
        cost_dict = self.unf.costs()
        if self.TO_SVOS: cost_dict['svo'] = self.svo.agent.dollar_cost()
        return cost_dict

    # end overrides

    def solve(self):

        self.resume()

        def step(g, gs, d):
            if g in gs: return
            self.persist()
            if d >= self.lim:
                if self.appraise(g, gs):
                    self.facts[g] = True
                    yield g, gs
            else:
                hs = []
                for h, bs in self.new_clause(g, gs):
                    if not self.appraise(h, (g, gs)): continue
                    hs.append(h)
                    trace = h, (g, gs)
                    self.clauses[g].append([h])
                    self.clauses[h].append(bs)
                    for b in bs:
                        yield from step(b, trace, d + 1)
                if self.strict and len(hs) > 1: self.clauses['false'].append(hs)

        for gs in step(self.initiator, (), 0):
            yield list(reversed(to_list(gs)))

        self.persist()

        for fact in self.facts: self.clauses[fact].append([])
        css = [(h, bs) for (h, bss) in self.clauses.items() for bs in bss]
        self.logic_model = qprove(css, goal=self.initiator)
        self.trim_clauses()
        self.save_results()
        if self.TO_SVOS:
            self.svo = SvoMaker(self.name)

    def save_results(self):
        pro_name = f'{self.OUT}{self.name}_{self.pname}_{self.lim}'
        mo_name = pro_name + "_model"
        json_name = pro_name + ".json"

        to_prolog(self.clauses, pro_name)
        to_json(self.clauses,json_name)

        if self.logic_model is None:
            self.logic_model = []
            tprint('\nNO MODEL ENTAILING:', self.initiator)
        else:
            tprint('\nMODEL:', len(self.logic_model), 'facts', '\n')
            for fact in self.logic_model: tprint(fact)
        save_model(self.initiator, self.logic_model, mo_name)

    def run(self):
        yield 'PROMPTER', self.prompter
        for r in self.solve():
            yield 'TRACE', r
        yield 'CLAUSES', dict(self.clauses)
        yield 'MODEL', self.logic_model
        if self.TO_SVOS: yield 'SVOS', self.svo.to_svos(self.logic_model, self.clauses)
        yield 'COSTS', self.costs()

    # -------- begin overrides -------------

    def resume(self):
        self.unf.resume()

    def persist(self):
        self.unf.persist()

    def appraise(self, g, _trace):
        """
        to be overriden by refiners
        """
        return True

    def trim_clauses(self):
        """
        can be overriden : for now, only atoms in the model
        will be kept
        """
        clauses = defaultdict(list)
        if self.logic_model is not None:
            model = set(self.logic_model)
            clauses = defaultdict(list)
            for (h, bss) in self.clauses.items():
                for bs in bss:
                    ok = all(b in model for b in bs)
                    if ok: clauses[h].append(bs)
        self.clauses = clauses


def run_explorer(goal=None, prompter=None, lim=None):
    assert None not in (prompter, goal, lim)
    r = AndOrExplorer(initiator=goal, prompter=prompter, lim=lim)
    seen = dict()
    for a in r.solve():
        print('\nTRACE:')
        # a=sorted(a)
        for x in a: print(x)
        print()
        a = tuple(a)
        if a in seen:
            print('--------------SEEN:', a)
        else:
            seen[a] = True

        if len(a) != len(set(a)):
            print('--------------REPEATED:', a)
    r.unf.persist()
    c1 = r.unf.or_.dollar_cost()
    c2 = r.unf.and_.dollar_cost()
    print('COSTS in $:', {'and': c1, 'or': c2, 'total': c1 + c2})
    return True


def quote(x):
    x=x.replace('\\','').replace("'","_")
    return "'" + x + "'"


def show_clauses(clauses):
    buf = []

    def p(x):
        buf.append(x)

    for h, bss in clauses.items():
        if bss == []:
            p(f"'{h}'.\n")
            break

        for i, bs in enumerate(bss):
            if i == 0:
                end = ":-" if bs else "."
                p(f"'{h}'{end}\n")

            for j, b in enumerate(bs):
                b = f"'{b}'"
                if j + 1 == len(bs):
                    # print('!!!!', b)
                    b = b + ("." if i + 1 == len(bss) else ";")
                else:
                    b = b + ","
                p("    " + b + "\n")

    return "".join(buf)


class SvoMaker:
    def __init__(self, topic, min_words=2):
        prompter = hyper_prompter
        pname = prompter['name']
        tname = topic.replace(' ', '_').lower()
        self.topic=topic
        self.agent = Agent(f'{tname}_{pname}')
        self.agent.set_pattern(prompter['hyper_p'])
        self.min_words = min_words
        PARAMS()(self)

    def to_svo(self, sentence):
        # print('<<<',sentence)
        answer = self.agent.ask(g=sentence, context=self.topic)

        return sentence,'is',answer

    def to_svos(self, facts, clauses):

        # jpp(clauses)
        svos = []
        self.resume()
        for fact in facts:
            svo = self.to_svo(fact)
            if not svo and fact:
                svos.append(('it', 'is assumed', fact))
                continue
            svos.append(svo)
            # print("SVO:", svo)
            self.agent.spill()
            s, v, o = svo

            body = clauses[fact]
            for ors in body:
                assert isinstance(ors, list), ors
                for and_ in ors:
                    svos.append((fact, ':', and_))

        self.persist()
        # jpp(svos)
        return svos

    def resume(self):
        return self.agent.resume()

    def persist(self):
        return self.agent.persist()

    def costs(self):
        return {"svo": self.agent.dollar_cost()}


def show_svos(svos):
    return json.dumps(svos, indent=2)


def vis_svos(svos, fname='rel_graph', show=True):
    return visualize_rels(svos, fname=fname, show=show)


def show_model(facts):
    buf = []
    for fact in facts:
        buf.append(f"'{fact}'.")
    return "\n".join(buf)


def save_model(goal, facts, fname, suf='.pro'):
    path = fname + suf
    ensure_path(path)
    with open(path, 'w') as f:
        print(f'% MODEL: {len(facts)} facts', file=f)
        for fact in facts:
            line = quote(fact) + "."
            if fact == goal: line = line + "%" + (10 * " ") + "<==== initiator !"
            print(line, file=f)


def to_prolog(clauses, fname, neck=":-", suf='.pro'):
    suf = '.nat' if neck == ":" else suf

    path = fname + suf
    ensure_path(path)
    with open(path, 'w') as f:
        print('% CLAUSES:', file=f)
        rule_heads=set()
        for h, bss in clauses.items():
            for bs in bss:
                if bs == [] or bs == ['fail']: continue
                body = ',\n    '.join(map(quote, bs)) + "."
                print(quote(h), neck + '\n    ' + body, file=f)
                rule_heads.add(h)
        for h, bss in clauses.items():
            if bss == []:
                if h in rule_heads: continue
                print(quote(h) + ".", file=f)
            for bs in bss:
                if bs == []:
                    if h in rule_heads: continue
                    print(quote(h) + ".", file=f)
                elif bs == ['fail']:
                    if h in rule_heads: continue
                    print(quote(h), neck + ' ' + 'fail.', file=f)


def to_context(trace, topgoal):
    if not trace: return topgoal
    context = ".\n".join(reversed(to_list(trace))) + ".\n"
    # print('!!!! CONTEXT:',context, '!!!!\n')
    return context


def test_svo(sent="The black cat sits on the shiny white mat"):
    m = SvoMaker(topic='test')
    print(sent)
    print(m.to_svo(sent))
    print()


if __name__ == "__main__":
    test_svo()
    test_svo('The  elephant in the room')
    test_svo("Jason's dog")
    test_svo('The  unexpected end of the blue water world')

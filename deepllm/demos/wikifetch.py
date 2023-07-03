import wikipediaapi
from fast_sentence_segment import segment_text
from deepllm.params import PARAMS,ensure_path

import logging
logger=logging.getLogger()
logger.setLevel(logging.CRITICAL)

def page2text(page_name, lang='en'):
    print('PROCESSING WIKI FOR:',page_name)
    wiki_wiki = wikipediaapi.Wikipedia(
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    page = wiki_wiki.page(page_name)
    text = page.text
    if not text:
        print('NO WIKY ENTRY FOR:', page_name)
        return None
    # print(text)
    sents = segment_text(text, flatten=True)
    cleans = []
    good = "'~:;=/*()[]{},.?!-+" + '"'
    keep = "$%"
    for s in sents:
        # print(s);continue
        for g in good:
            s = s.replace(g, ' ')
        for g in keep: s = s.replace(g, ' ' + g + ' ')
        xs = s.split()
        raw = len(xs)
        xs = [x.strip() for x in xs if x.isalnum() or x in keep]
        cleaned = len(xs)
        if cleaned > 5 and cleaned / raw > 0.8 and len(xs[-1]) > 3 and xs[0] == xs[0].capitalize():
            clean = " ".join(xs)
            cleans.append(clean + ".")
    fname = page_name.lower().replace(' ', '_').replace('.', '_')
    if cleans:
        path=f'{PARAMS().DATA}/{fname}.txt'
        ensure_path(path)
        ground_truth_file = path
        with open(ground_truth_file, 'w') as f:
            for x in cleans:
                print(x, file=f)
        return ground_truth_file
    else:
        print('NO FILE GENERATED FOR:', page_name)
        return None


def run_wikifetch():
    page2text('Logic programming')
    page2text('Computational thinking')
    page2text('Artificial general intelligence')
    page2text('Expansion of the universe')
    page2text('No such page')


if __name__ == "__main__":
    pass
    run_wikifetch()


from nltk import Tree
from nltk.corpus.reader import CategorizedBracketParseCorpusReader
from nltk.corpus.util import LazyCorpusLoader

from collections import defaultdict

ptb_train = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/((0[2-9])|(1\d)|(2[0-1]))/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')
ptb_test = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/23/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')
ptb_devel = LazyCorpusLoader('ptb', CategorizedBracketParseCorpusReader, r'wsj/24/wsj_.*.mrg', cat_file='allcats.txt', tagset='wsj')

def extract_sents(corpus, target):
    with open(target + "_tokens.txt", "w") as f:
        with open(target + "_tags.txt", "w") as g:
            for sent in corpus.parsed_sents():
                tokens = [tup[0] for tup in sent.pos()]
                tags = [tup[1] for tup in sent.pos()]
                assert len(tokens) == len(tags)

                for i in range(len(tokens)-1):
                    f.write(tokens[i])
                    f.write('\t')
                f.write(tokens[-1])
                f.write('\n')

                for i in range(len(tags)-1):
                    g.write(tags[i])
                    g.write('\t')
                g.write(tags[-1])
                g.write('\n')


def extract_tagset(corpus):
    tags = set()
    for sent in corpus.parsed_sents():
        for (token, tag) in sent.pos():
            tags.add(tag)

    with open("tagset.txt", "w") as f:
        for tag in sorted(tags):
            f.write(tag + "\n")

extract_sents(ptb_train, "train")
extract_sents(ptb_test, "test")
extract_sents(ptb_devel, "devel")
extract_tagset(ptb_train)

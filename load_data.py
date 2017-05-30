import numpy as np


def load_sents(tokens_file, tags_file):
    '''Load tokens and tags from files'''
    tokens = list()
    tags = list()
    with open(tokens_file) as f:
        with open(tags_file) as g:
            for line in f:
                line = line.strip().split('\t')
                tokens.append(line)
            for line in g:
                line = line.strip().split('\t')
                tags.append(line)
                
    assert len(tokens) == len(tags)
    sents = [(tokens[i], tags[i]) for i in range(len(tokens))]
    
    return sents



def load_embeddings(filename, dim):
    '''Load word embeddings of the specified dimension from a file'''
    token_to_ix = dict()
    ix_to_token = dict()
    embeds = list()

    with open(filename) as f:
        i = 0
        for line in f:
            line = line.split(' ')
            token = line[0]
            vals = list(map(float, line[1:dim+1]))
            token_to_ix[token] = i
            ix_to_token[i] = token
            embeds.append(vals)
            i += 1

    # Add traces etc.
    token_to_ix['TRACE'] = i
    ix_to_token[i] = 'TRACE'
    embeds.append([np.random.uniform(low=-0.1, high=0.1) for _ in range(dim)])
    i += 1 

    token_to_ix['NUMBER'] = i
    ix_to_token[i] = 'NUMBER'
    embeds.append([np.random.uniform(low=-0.1, high=0.1) for _ in range(dim)])
    i += 1

    token_to_ix['UNK'] = i
    ix_to_token[i] = 'UNK'
    embeds.append([np.random.uniform(low=-0.1, high=0.1) for _ in range(dim)])
    i += 1

    return token_to_ix, ix_to_token, np.array(embeds)



def load_tagset(tagset_file):
    i = 0
    tag_to_ix = dict()
    ix_to_tag = dict()
    with open(tagset_file) as f:
        for line in f:
            tag = line.strip()
            tag_to_ix[tag] = i
            ix_to_tag[i] = tag
            i += 1

    return tag_to_ix, ix_to_tag



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



def capitalization(string):
    if string.isupper():
        return 2
    elif string[0].isupper():
        return 1
    else:
        return 0



def normalize_sent(sent):
    sent = [token.lower() for token in sent]
    normalized_sent = list()
    for token in sent:
        if token == "-lrb-" or token == "-lcb-":
            token = "("
        elif token == "-rrb-" or token == "-rcb-":
             token = ")"
        elif token.startswith("*") or token == "0":
            token = "TRACE"
        elif is_number(token):
            token = "NUMBER"
        normalized_sent.append(token)
        
    assert len(normalized_sent) == len(sent)
    return normalized_sent



def sent_as_numbers(sent, token_to_ix, tag_to_ix):
    tokens, tags = sent
    token_numbers = [token_to_ix[token] if token in token_to_ix else token_to_ix["UNK"] for token in normalize_sent(tokens)]
    caps = list(map(capitalization, tokens))
    tag_numbers = [tag_to_ix[tag] for tag in tags]

    assert len(token_numbers) == len(caps) == len(tag_numbers)
    return (token_numbers, caps, tag_numbers)



def get_batches():
    pass



def load_all_data(tokens_file, tags_file, tagset_file, embeddings_file, embeddings_dim):
    # TODO batching (?)
    sents = load_sents(tokens_file, tags_file)
    token_to_ix, ix_to_token, embeds = load_embeddings(embeddings_file, embeddings_dim)
    tag_to_ix, ix_to_tag = load_tagset(tagset_file)
    
    return [sent_as_numbers(sent, token_to_ix, tag_to_ix) for sent in sents], token_to_ix, ix_to_token, tag_to_ix, ix_to_tag, embeds

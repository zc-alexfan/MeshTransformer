import torch


def preprocess_word(w):
    w = w.lower()
    w = w.strip()
    return w


def numerize_phrase_list(
    phrase_list, word2idx, process_word_cb=preprocess_word, unk_token="<unk>"
):
    """
    Given a list of lists of tokens, numerize it into a list of LongTensor.
    Note: It does not add SOS and EOS and padding here.
    Inputs:
        phrase_list: e.g., [['hello', 'World'], ['Another']]
        word2idx: {'token1': 1, 'token2':2}
        (Optional) process_word_cb takes in a str and return a processed str.
    Outputs:
        [LongTensor, LongTensor]
    """
    assert isinstance(phrase_list, list)
    assert isinstance(word2idx, dict)
    phrase_indices_list = []
    for phrase in phrase_list:
        phrase_indices = []
        for w in phrase:
            w = process_word_cb(w)
            if w not in word2idx.keys():
                w = unk_token
            phrase_indices.append(word2idx[w])
        phrase_indices_list.append(torch.LongTensor(phrase_indices))
    return phrase_indices_list


def pad_phrase_indices(target_phrase_indices, eos_token=None):
    """
    Given a list of LongTensor, pad them into a matrix.
    eos_token: add a EOS_TOKEN before padding it if it is not None
    e.g., [LongTensor([1, 2]), LongTensor([3]), LongTensor([4])]
    It will return LongTensor with shape = (num_phrases, max_len)
    """
    assert isinstance(target_phrase_indices, list)
    assert isinstance(
        target_phrase_indices[0], (torch.LongTensor, torch.cuda.LongTensor)
    )

    batch_size = len(target_phrase_indices)
    assert batch_size > 0
    max_len = max([len(phrase_indices) for phrase_indices in target_phrase_indices])

    if eos_token is not None:
        max_len += 1
    padded_target = torch.zeros((batch_size, max_len), dtype=torch.long)

    # for each phrase
    for phrase_i, phrase_indices in enumerate(target_phrase_indices):
        padded_target[phrase_i, : len(phrase_indices)] = phrase_indices
        if eos_token is not None:
            padded_target[phrase_i, len(phrase_indices)] = eos_token
    return padded_target

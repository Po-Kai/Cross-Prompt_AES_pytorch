import re


def text_preprocess(text):
    text = text.strip()
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"-?\d+(\.\d+)?", "[NUM]", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace(u'"', u"")
    text = text.replace("�", "")
    text = re.sub(r"@([A-Z]+)[0-9]+", r"[\1]", text)  # @CAP1 -> [CAP]
    
    if "..." in text:
        text = re.sub(r"\.{3,}(\s+\.{3,})*", "...", text)
    if "??" in text:
        text = re.sub(r"\?{2,}(\s+\?{2,})*", "?", text)
    if "!!" in text:
        text = re.sub(r"\!{2,}(\s+\!{2,})*", "!", text)
    return text


def tokenize_to_sentences(text, max_sent_len, tokenizer):
    sents = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s", text)
    processed_sents = []
    for sent in sents:
        if re.search(r"(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)", sent):
            s = re.split(r"(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)", sent)
            ss = " ".join(s)
            ssL = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s", ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sent_len, tokenizer)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens


def shorten_sentence(sent, max_sent_len, tokenizer):
    tokenized_sents = []
    sent = sent.strip()
    tokens = tokenizer.tokenize(sent)
    if len(tokens) > max_sent_len:
        split_keywords = [
            "because", "but", "so", "You", "He", 
            "She", "We", "It", "They", "Your", 
            "His", "Her"
        ]
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sent_len
            num = int(round(num))
            k_indexes = [(i+1)*max_sent_len for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        for token in processed_tokens:
            if len(token) > max_sent_len:
                num = len(token) / max_sent_len
                num = int(np.ceil(num))
                s_indexes = [(i+1)*max_sent_len for i in range(num)]

                len_s = len(s_indexes)
                tokenized_sents.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    tokenized_sents.append(token[s_indexes[j]:s_indexes[j+1]])
                tokenized_sents.append(token[s_indexes[-1]:])

            else:
                tokenized_sents.append(token)
        
    else:
        tokenized_sents = [tokens]

    return [" ".join(tokenized_sent) for tokenized_sent in tokenized_sents]
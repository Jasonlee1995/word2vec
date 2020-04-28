import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.manual_seed(0)


# Make Dataset for Each Model
def make_dataset(mode, sentences, vocab2id, window_size):
    data = list()
    for sentence in sentences:
        ids = [vocab2id[vocab] for vocab in sentence]
        for i in range(len(ids)):
            if mode == 'cbow':
                X, y = list(), ids[i]
                for j in range(max(i-window_size, 0), min(i+1+window_size, len(sentence))):
                    if j != i: X.append(ids[j])
                data.append((torch.LongTensor(X), y))
            elif mode == 'skip-gram':
                X = ids[i]
                random_window = torch.LongTensor(1).random_(1, window_size+1).item()
                for j in range(max(i-random_window, 0), min(i+1+random_window, len(sentence))):
                    if j != i:
                        y = ids[j]
                        data.append((X, y))
    if mode == 'cbow':
        return DataLoader(data, pin_memory=True, num_workers=0)
    elif mode == 'skip-gram':
        return DataLoader(data, pin_memory=True, num_workers=0)

# Return top 15 nearest words for target word of model (euclidean distance)
def word_euclidean(word2vec, target, vocab_set, vocab2id):
    word2vec.model.to('cpu')
    target_embed = word2vec.model.embeddings(torch.LongTensor([[vocab2id[target]]]))
    target_similar = dict()

    for vocab in vocab_set:
        if vocab != target:
            vocab_embed = word2vec.model.embeddings(torch.LongTensor([[vocab2id[vocab]]]))
            similarity = torch.dist(target_embed, vocab_embed, 2).item()
            if len(target_similar) < 15:
                target_similar[round(similarity, 6)] = vocab
            elif min(target_similar.keys()) > similarity:
                del target_similar[min(target_similar.keys())]
                target_similar[round(similarity, 6)] = vocab
    
    return sorted(target_similar.items(), reverse=False)


# Return top 15 nearest words for target word of model (cosine similarity)
def word_cosine(word2vec, target, vocab_set, vocab2id):
    word2vec.model.to('cpu')
    target_embed = word2vec.model.embeddings(torch.LongTensor([[vocab2id[target]]]))
    target_similar = dict()

    for vocab in vocab_set:
        if vocab != target:
            vocab_embed = word2vec.model.embeddings(torch.LongTensor([[vocab2id[vocab]]]))
            similarity = F.cosine_similarity(*target_embed, *vocab_embed).item()
            if len(target_similar) < 15:
                target_similar[round(similarity, 6)] = vocab
            elif min(target_similar.keys()) < similarity:
                del target_similar[min(target_similar.keys())]
                target_similar[round(similarity, 6)] = vocab
    
    return sorted(target_similar.items(), reverse=False)

import math

def precision_at_k(recs, truth, k):
    if k == 0:
        return 0.0
    recs_k = recs[:k]
    hits = sum(1 for r in recs_k if r in truth)
    return hits / k

def recall_at_k(recs, truth, k):
    if not truth:
        return 0.0
    recs_k = recs[:k]
    hits = sum(1 for r in recs_k if r in truth)
    return hits / len(truth)

def ndcg_at_k(recs, truth, k):
    recs_k = recs[:k]
    dcg = 0.0
    for i, r in enumerate(recs_k, start=1):
        if r in truth:
            dcg += 1.0 / math.log2(i + 1)
    # ideal DCG
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0

import random

class MSMARCO:
    def __init__(self,
                 queries:dict,
                 qrels:dict,
                 passages:dict,
                 batch_size:int=256,
                 batch_negatives:int=0,
                 shuffle:bool=True):

        self.queries=queries
        self.qrels=qrels
        self.passages=passages
        self.pids=set(passages.keys())
        self.qids=list(queries.keys())
        self.batch_size=batch_size
        self.batch_negatives=batch_negatives
        self.suffle=shuffle


    def fetch(self):
        
        batch=[]
        remaining_qids=self.qids.copy()
        unavailable_pids=set()

        while len(batch) < self.batch_size:
            qid = remaining_qids.pop(random.randrange(len(remaining_qids)))
            pos_pids=[k for k, v in self.qrels[qid].items() 
                    if k not in unavailable_pids and v > 0]

            if pos_pids and self.batch_negatives>0:

                pid=random.choice(pos_pids)
                passage=self.passages[pid]['text']
                query=self.queries[qid]
                neg_pids=random.sample([k for k, v in self.qrels[qid].items() 
                                        if v < 0], self.batch_negatives)
                passages=[passage]+[self.passages[neg_pid]['text'] 
                                    for neg_pid in neg_pids]
                batch.append((query, passages))

            elif pos_pids:
                pid=random.choice(pos_pids)
                passage=self.passages[pid]['text']
                query=self.queries[qid] 
                batch.append((query, [passage]))
            
            unavailable_pids.update(pos_pids)

        return(batch)





    

gold_path = 'ChineseWordSegment/data/baike/baike.segs.pseudo'
pred_path = 'ChineseWordSegment/predict/mws/bert.baike.test'

golds_word, golds_no_word, preds = [], [], []
with open(gold_path,'r') as f:
    lines = [line.strip() for line in f]
    golds_word = lines[1::2]
    golds_word = [eval(item) for item in golds_word]

with open(pred_path,'r') as f:
    lines = [line.strip() for line in f]
    preds = lines[1::2]
    preds = [eval(item) for item in preds]

class SegF1Metric():

    def __init__(self, eps, pred_words, gold_words):
        self.pred_words = pred_words
        self.gold_words = gold_words
        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps
        self.p = 0.0
        self.r = 0.0
        self.f = 0.0
        self.count(self.pred_words, self.gold_words)
        

    def have_intersection(self, tuple1, tuple2):
        return max(tuple1[0], tuple2[0]) < min(tuple1[1], tuple2[1])
        # return tuple1[1] > tuple2[0] and tuple1[0] < tuple2[1]

    def count(self, pred_words, gold_words):
        for pred, gold in zip(pred_words,gold_words):
            self.tp += len(set(pred) & set(gold))
            self.gold += len(gold)
            for item1 in pred:
                flag = False
                for item2 in gold:
                    if self.have_intersection(item1,item2):
                        flag = True
                if flag:
                    self.pred += 1
        self.p = self.tp / (self.pred + self.eps)
        self.r = self.tp / (self.gold + self.eps)
        self.f = 2 * self.tp / (self.pred + self.gold + self.eps)

if __name__ == '__main__':
    metric_span = SegF1Metric(eps=1e-8,pred_words=preds,gold_words=golds_word)
    print(metric_span.p)
    print(metric_span.r)
    print(metric_span.f)



    

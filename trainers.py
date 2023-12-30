from transformers import Trainer


class AESTrainer(Trainer):

    def __init__(self, evaluator, *args, **kwargs):
        super(AESTrainer, self).__init__(*args, compute_metrics=self.compute_metrics, **kwargs)
        self.evaluator = evaluator
        self.dev_size = len(self.evaluator.dev_dataset)
        
    def compute_metrics(self, p):
        preds, _ = p.predictions[1], p.label_ids
        dev_preds = preds[:self.dev_size]
        test_preds = preds[self.dev_size:]
        results = self.evaluator.evaluate(dev_preds, test_preds, self.state.epoch)
        return {"kappa": results["test_kappa_mean"]}
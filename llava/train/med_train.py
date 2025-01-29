import os
import sys
from unittest import mock

# from llava.train.train import train
from llava.train.med_train_util import train

from llava.train.transformer_normalize_monkey_patch import (
    _save_checkpoint,
    compute_loss,
    patched_normalize,
    training_step,
)

def __len__(self):
    return len(self.batch_sampler)


def __iter__(self):
    return self.batch_sampler.__iter__()

if __name__ == "__main__":
    with (
        mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
        # mock.patch("accelerate.data_loader.BatchSamplerShard.__len__", new=__len__),
        # mock.patch("accelerate.data_loader.BatchSamplerShard.__iter__", new=__iter__),
        # mock.patch("transformers.trainer.Trainer._save_checkpoint", new=_save_checkpoint),
        # mock.patch("transformers.trainer.Trainer.compute_loss", new=compute_loss),
        # mock.patch("transformers.trainer.Trainer.training_step", new=training_step),
    ):
        
        train()
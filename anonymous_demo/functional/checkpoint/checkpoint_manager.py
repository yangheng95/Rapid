import os
from findfile import find_file

from anonymous_demo.core.tad.prediction.tad_classifier import TADTextClassifier
from anonymous_demo.utils.demo_utils import retry


class CheckpointManager:
    pass


class TADCheckpointManager(CheckpointManager):
    @staticmethod
    @retry
    def get_tad_text_classifier(checkpoint: str = None,
                                eval_batch_size=128,
                                **kwargs):

        tad_text_classifier = TADTextClassifier(checkpoint, eval_batch_size=eval_batch_size, **kwargs)
        return tad_text_classifier

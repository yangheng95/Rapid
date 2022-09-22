from anonymous_demo import TADCheckpointManager

from textattack.model_args import DEMO_MODELS
from textattack.reactive_defense.reactive_defender import ReactiveDefender


class TADReactiveDefender(ReactiveDefender):
    """ Transformers sentiment analysis pipeline returns a list of responses
        like

            [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

        We need to convert that to a format TextAttack understands, like

            [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, ckpt='tad-sst2', **kwargs):
        super().__init__(**kwargs)
        self.tad_classifier = TADCheckpointManager.get_tad_text_classifier(checkpoint=DEMO_MODELS[ckpt],
                                                                           auto_device=True)

    def reactive_defense(self, text, **kwargs):
        res = self.tad_classifier.infer(text, defense='pwws', print_result=False, **kwargs)
        return res

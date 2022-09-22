import anonymous_demo.core.tad.classic.__bert__.models


class BERTTADModelList(list):
    TADBERT = anonymous_demo.core.tad.classic.__bert__.TADBERT

    def __init__(self):
        model_list = [self.TADBERT]
        super().__init__(model_list)

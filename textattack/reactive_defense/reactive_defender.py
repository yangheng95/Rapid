from abc import ABC, abstractmethod

from textattack.shared.utils import ReprMixin


class ReactiveDefender(ReprMixin, ABC):
    def __init__(self, **kwargs):
        pass

    def reactive_defense(self, **kwargs):
        pass

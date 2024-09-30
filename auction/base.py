from __future__ import annotations
from abc import ABC
from logging import Logger
from typing import Generator, Type, Dict

from auction.utils import console_logger


class Strategy(ABC):
    """
    Base class for creating objects of subclasses from class name and parameters
    """

    @property
    def class_name(self) -> str:
        return self._get_class_name()

    @classmethod
    def _get_class_name(cls) -> str:
        return cls.__name__

    @classmethod
    def _get_subclasses(cls) -> Generator[Type[Strategy], None, None]:
        """
        Get all subclasses of a class
        :return: List of all available subclasses
        """
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def get_by_strategy(cls, strategy: str) -> Type[Strategy]:
        """
        Get subclass type from class_name or strategy name
        :param strategy: strategy name or subclass name
        :return: strategy type
        """
        if cls.__name__ == 'Strategy':
            raise RuntimeError('Method is only available for subclasses of Strategy')

        subclasses_map = {}
        for subclass in cls._get_subclasses():
            # in some cases we have `strategy` as a class_attr instead class name
            if hasattr(subclass, "strategy"):
                subclasses_map[subclass.strategy] = subclass
            else:
                subclasses_map[subclass._get_class_name()] = subclass

        if strategy not in subclasses_map:
            raise NameError(f'Strategy {strategy} not found, available strategies are '
                            f'{list(subclasses_map.keys())} in all subclasses of a class {cls.__name__}')
        return subclasses_map[strategy]

    @classmethod
    def create_subclass_object(cls, strategy: str, **init_kwargs) -> Strategy:
        """
        Create object directly from `strategy` (or class name) and its init params
        :param strategy: desired class name or strategy
        :param init_kwargs: init arguments for class creation
        :return: subclass object
        """
        cls_type = cls.get_by_strategy(strategy)
        obj = cls_type(**init_kwargs)
        return obj


class StrategyAuction(Strategy):
    """
    Base class for dynamic subclass search with init.
    You can set up params and logger here
    """

    def __init__(self, params: Dict, logger: Logger = console_logger(''), **kwargs):
        """
        Strategy init interface
        :param params: specific parameters for strategy type
        :param logger: logger
        :param kwargs:
        """
        self.params = params
        self.logger = logger
"""Provide a set of classes to abstractly define filters on data sets."""

from abc import ABC, abstractmethod


class AbstractFilter(ABC):
    @abstractmethod
    def apply(self, *args, **kwargs):
        pass


class EqualFilter(AbstractFilter):
    def __init__(self, target):
        self.target = target

    def apply(self, arr):
        return (arr == self.target)


class InRangeFilter(AbstractFilter):
    def __init__(self, low: float = -np.inf, high: float = np.inf,
                 low_inc: bool = True, high_inc: bool = True):
        self.low = low
        self.high = high
        self.low_inc = low_inc
        self.high_inc = high_inc

    def apply(self, arr):
        if self.low_inc:
            if self.high_inc:
                return (self.low <= arr) & (arr <= self.high)
            else:
                return (self.low <= arr) & (arr < self.high)
        else:
            if self.high_inc:
                return (self.low < arr) & (arr <= self.high)
            else:
                return (self.low < arr) & (arr < self.high)


class IsNAFilter(AbstractFilter):
    def apply(self, arr):
        return np.isnan(arr)
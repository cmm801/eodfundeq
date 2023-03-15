"""Provide a set of classes to abstractly define filters on data sets."""

import numpy as np

from abc import ABC, abstractmethod


class AbstractFilter(ABC):
    def __init__(self, instance, property_name, *args, **kwargs):
        self.instance = instance
        self.property_name = property_name

    @property
    def property_values(self):
        return getattr(self.instance, self.property_name)

    @abstractmethod
    def get_mask(self, *args, **kwargs):
        pass


class EqualFilter(AbstractFilter):
    def __init__(self, instance, property_name, target_value):
        super(EqualFilter, self).__init__(instance, property_name)
        self.target_value = target_value

    def get_mask(self):
        return (self.property_values == self.target_value)


class InRangeFilter(AbstractFilter):
    def __init__(self, instance, property_name, low: float = -np.inf, high: float = np.inf,
                 low_inc: bool = True, high_inc: bool = True):
        super(InRangeFilter, self).__init__(instance, property_name)
        self.low = low
        self.high = high
        self.low_inc = low_inc
        self.high_inc = high_inc

    def get_mask(self):
        if self.low_inc:
            if self.high_inc:
                return (self.low <= self.property_values) & \
                       (self.property_values <= self.high)
            else:
                return (self.low <= self.property_values) & \
                       (self.property_values < self.high)
        else:
            if self.high_inc:
                return (self.low < self.property_values) & \
                       (self.property_values <= self.high)
            else:
                return (self.low < self.property_values) & \
                       (self.property_values < self.high)


class EntireColumnInRangeFilter(InRangeFilter):
    def __init__(self, instance, property_name, low: float = -np.inf, high: float = np.inf,
                 low_inc: bool = True, high_inc: bool = True, ignore_na: bool = True):
        super(EntireColumnInRangeFilter, self).__init__(instance, property_name,
            low=low, high=high, low_inc=low_inc, high_inc=high_inc)
        self.ignore_na = ignore_na

    def get_mask(self):
        super_mask = super().get_mask()
        if self.ignore_na:
            super_mask |= np.isnan(self.property_values)
        return np.all(super_mask, axis=0).reshape(1, -1)


class EntireRowInRangeFilter(InRangeFilter):
    def __init__(self, instance, property_name, low: float = -np.inf, high: float = np.inf,
                 low_inc: bool = True, high_inc: bool = True, ignore_na: bool = True):
        super(EntireRowInRangeFilter, self).__init__(instance, property_name,
            low=low, high=high, low_inc=low_inc, high_inc=high_inc)
        self.ignore_na = ignore_na

    def get_mask(self):
        super_mask = super().get_mask()
        if self.ignore_na:
            super_mask |= np.isnan(self.property_values)
        return np.all(super_mask, axis=1).reshape(-1, 1)


class IsNotNAFilter(AbstractFilter):
    def get_mask(self):
        return ~np.isnan(self.property_values)


class IsNAFilter(AbstractFilter):
    def get_mask(self):
        return np.isnan(self.property_values)
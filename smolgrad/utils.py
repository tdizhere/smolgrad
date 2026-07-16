from engine import Value
import math
import random
def mean(values):
    return sum(values, Value(0)) / len(values)

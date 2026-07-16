from engine import Value
import math
class MSELoss():
    def __call__(self,prediction,target):
        return sum(((prediction - target) ** 2 for prediction, target in zip(prediction, target)),Value(0))
class L1Loss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, prediction, target):
        loss = sum(
            (abs(p - t) for p, t in zip(prediction, target)),
            Value(0)
        )

        if self.reduction == "sum":
            return loss
        elif self.reduction == "mean":
            return loss / len(prediction)
        else:
            raise ValueError("Unknown reduction")

class BCELoss:

    def __call__(self, prediction, target):

        eps = 1e-7

        loss = sum(
            (
                -(
                    t * p.clamp(eps,1-eps).log()
                    +
                    (1-t) * (1-p).clamp(eps,1-eps).log()
                )
                for p,t in zip(prediction,target)
            ),
            Value(0)
        )

        return loss / len(prediction)
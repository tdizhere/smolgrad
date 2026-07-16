from engine import Value
from nn import *
from loss import MSELoss
from utils import *
from optim import *
from dataset import *


def accuracy(predictions, targets):

    correct = 0

    for p, y in zip(predictions, targets):

        pred = 1 if p.data > 0.5 else 0

        if pred == y.data:
            correct += 1

    return correct / len(targets)



def main():

    # dataset making 

    xs, ys = linear()
    print("Dataset")
    print("----------------")
    print("Samples:", len(xs))
    print("Features:", len(xs[0]))



    # define model

    model = Sequential(

        Linear(2,16),
        Tanh(),

        Linear(16,16),
        Tanh(),

        Linear(16,1)

    )


    print("\nModel")
    print("----------------")
    print(model)

    print(
        "Parameters:",
        model.num_parameters()
    )



    # initialize loss and optim

    criterion = MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.1)



    # train

    epochs = 50


    for epoch in range(epochs):


        predictions = [
            model(x)
            for x in xs
        ]


        loss = criterion(
            predictions,
            ys
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        if epoch % 1 == 0:

            acc = accuracy(
                predictions,
                ys
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.data:.5f} | "
                f"Accuracy {acc*100:.2f}%"
            )



    # test the predictions now 

    print("\nPredictions")
    print("----------------")


    for x,y in zip(xs[:10],ys[:10]):

        pred = model(x)

        print(
            f"Input: {[v.data for v in x]} "
            f"Target: {y.data:.0f} "
            f"Prediction: {pred.data:.4f}"
        )



if __name__ == "__main__":
    main()
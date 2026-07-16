from engine import Value
import random 
class Module:
    def num_parameters(self):
        return len(self.parameters())
    def parameters(self):
        return []
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    def step(self,lr=0.01):
        for p in self.parameters():
            p.data -= lr * p.grad

class Neuron(Module):
    def __init__(self,nin,activation=None):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
        self.activation = activation
    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        if self.activation is None:
            return act

        return self.activation(act)
    def parameters(self):
        return self.w+[self.b]
    def __repr__(self):
        return f"Neuron({len(self.w)},Activation={self.activation})"
class Layer(Module):
    def __init__(self,nin,nout,**kwargs):
        self.neurons = [Neuron(nin,**kwargs) for _ in range(nout)]
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    def parameters(self):
        
        return [p for n in self.neurons for p in n.parameters()]
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
class MLP(Module):

    def __init__(self, nin, nouts,activation=None):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation=activation if i != len(nouts)-1 else None) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
class Linear(Module):

    def __init__(self, nin, nout, bias=True):
        self.neurons = [
            Neuron(nin, activation=None)
            for _ in range(nout)
        ]

        if not bias:
            for n in self.neurons:
                n.b = Value(0, requires_grad=False)

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [
            p
            for n in self.neurons
            for p in n.parameters()
        ]
class ReLU(Module):

    def __call__(self, x):
        if isinstance(x, list):
            return [v.relu() for v in x]
        return x.relu()
class Tanh(Module):

    def __call__(self, x):
        if isinstance(x, list):
            return [v.tanh() for v in x]
        return x.tanh()
class Sigmoid(Module):

    def __call__(self, x):
        if isinstance(x, list):
            return [v.sigmoid() for v in x]
        return x.sigmoid()
class LeakyReLU(Module):

    def __call__(self, x):
        if isinstance(x, list):
            return [v.leaky_relu() for v in x]
        return x.leaky_relu()
class Sequential(Module):

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]
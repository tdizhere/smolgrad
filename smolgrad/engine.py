import math

class Value:
    def __init__(self,data,_children=(),label="",requires_grad=True):
        self.requires_grad = requires_grad
        self.data = data
        self.label = label
        self._prev = tuple(_children)
        self.grad = 0.0
        self._backward = lambda: None
    def __repr__(self):
        return f"Value<Data={self.data}>"
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    def __pow__(self, other):
        assert isinstance(other,(int,float)) , "Value cant be raised to non-numeric power"
        out = Value(self.data ** other,(self,),f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    def __truediv__(self, other):
       return self * (other ** -1)
    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other
    def __rsub__(self, other):
        return self - other
    def __rmul__(self, other):
        return self * other
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self
    def __neg__(self):
        return self * -1
    def sqrt(self):
        out = Value(math.sqrt(self.data), (self,), "sqrt")

        def _backward():
            self.grad += out.grad * (1 / (2 * out.data))

        out._backward = _backward
        return out
    def __abs__(self):
        out = Value(abs(self.data), (self,), "abs")
        def _backward():
            if self.data > 0:
                self.grad += out.grad
            elif self.data < 0:
                self.grad -= out.grad
        # derivative at 0 is undefined; choose 0

        out._backward = _backward
        return self if self.data >= 0 else -self
    def backward(self):
        if self.requires_grad:
            # topological order all of the children in the graph — iterative version
            topo = []
            visited = set()
    
            def build_topo(root):
                stack = [(root, iter(root._prev))]
                visited.add(root)
                while stack:
                    node, children = stack[-1]
                    try:
                        child = next(children)
                        if child not in visited:
                            visited.add(child)
                            stack.append((child, iter(child._prev)))
                    except StopIteration:
                        topo.append(node)
                        stack.pop()
    
            build_topo(self)
    
            # go one variable at a time and apply the chain rule to get its gradient
            self.grad = 1
            for v in reversed(topo):
                v._backward()
    def tanh(self):
        x = self.data
       # t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        return out
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out
    def leaky_relu(self,alpha=0.01):
        out = Value(alpha*self.data if self.data < 0 else self.data, (self,), 'LeakyReLU')

        def _backward():
            self.grad += (1 if self.data >= 0 else alpha) * out.grad
        out._backward = _backward

        return out
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += out.grad *(1 / self.data)
        out._backward = _backward
        return out
    def sin(self):
        out = Value(math.sin(self.data), (self,), "sin")

        def _backward():
            self.grad += out.grad * math.cos(self.data)
            out._backward = _backward
        return out
    def cos(self):
        out = Value(math.cos(self.data), (self,), "cos")

        def _backward():
            self.grad += out.grad * -math.sin(self.data)
            out._backward = _backward
        return out
    def item(self):
        return float(self.data)
    def clamp(self, min_value, max_value):

        out = Value(
            min(max(self.data, min_value), max_value),
            (self,),
            "clamp"
        )


        def _backward():

            if min_value < self.data < max_value:
                self.grad += out.grad
            else:
                self.grad += 0


        out._backward = _backward

        return out   

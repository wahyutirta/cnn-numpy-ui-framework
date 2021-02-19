from relu import *
from fc import *
from Activation_Softmax import Activation_Softmax
from loss import *
from data import *

import numpy as np
from sklearn.metrics import accuracy_score
import timeit


class LENET5:
    """docstring forLENET5."""
    def __init__(self, t_input, t_output):

        fc1 = FC_LAYER(20, (2, 20), )#filename=b_dir+"fc6.npz")
        relu1 = RELU_LAYER()
        # Fully Connected-2
        fc2 = FC_LAYER(20, (20, 20), )#filename=b_dir+"fc8.npz")
        relu2 = RELU_LAYER()
        # Fully Connected-3
        output = FC_LAYER(3, (20, 3), )#filename=b_dir+"fc10.npz")
        #softmax = SOFTMAX_LAYER()
        softmax_cross = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.layers = [fc1, relu1, fc2, relu2, output, softmax_cross]

        self.X = t_input
        self.Y = t_output
        #self.Xv = v_input
        #self.Yv = v_output

    @staticmethod
    def feedForward(X, y,layers):
        """
        Computes final output of neural network by passing
        output of one layer to another.
        Input:
            X: Input
            layers: List of layers.
        Output:
            inp: Final output
        """
        y = y
        inp = X
        wsum = 0
        loss = 0
        for layer in layers:
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                loss, ws = layer.forward(inp,y)
                inp = layer.output
                #print("output shape ::", layer.output.shape)
                #print(layer.output)
            else:
                inp, ws = layer.forward(inp)
            wsum += ws
        return inp, loss ,wsum

    @staticmethod
    def backpropagation(Y, layers):
        """
        Computes final output of neural network by passing
        output of one layer to another.
        Input:
            Y: True output
            layers: List of layers.
        Output:
            grad: gradient
        """
        delta = Y
        for layer in layers[::-1]:
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                delta = layer.backward(layer.output,delta)
            else:
                delta = layer.backward(delta)

    @staticmethod
    def update_parameters(layers, batch_size, a, z, m):
        """
        Update weight parameters of each layer
        """
        for layer in layers:
            if isinstance(layer, FC_LAYER):
                layer.update_kernel(batch=batch_size, alpha=a, zeta=z, method=m)

    @staticmethod
    def loss_function(pred, t, **params):
        """
        Computes loss using cross-entropy method.
        Input:
            pred: Predicted output of network of shape (N, C)
            t: true output of shape (N, C)
            w_sum: sum of squares of all weight parameters for L2 regularization
        where,
            N: batch size
            C: Number of classes in the final layer
        Output:
            Loss or cost
        """
        w_sum = params.get("wsum", 0)
        #print("w_sum: ", w_sum)
        z = params.get("zeta", 0)

        assert t.shape == pred.shape
        #print("Shape: ", t.shape, z)
        epsilon = 1e-10
        return ((-t * np.log(pred + epsilon)).sum() + (z/2)*w_sum) / pred.shape[0]



    def lenet_train(self, **params):
        """
        Train the Lenet-5.
        Input:
            params: parameters including "batch", "alpha"(learning rate),
                    "zeta"(regularization parameter), "method" (gradient method),
                    "epochs", ...
        """
        batch  = params.get("batch", 300)             # Default 50
        alpha  = params.get("alpha", 0.01)            # Default 0.1
        zeta   = params.get("zeta", 0)               # Default 0 (No regularization)
        method = params.get("method", "adam")            # Default
        epochs = params.get("epochs", 1000)             # Default 4
        print("Training on params: batch=", batch, " learning rate=", alpha, " L2 regularization=", zeta, " method=", method, " epochs=", epochs)
        self.loss_history = []
        self.gradient_history = []
        self.valid_loss_history = []
        self.step_loss = []
        print(method)
        X_train = self.X
        Y_train = self.Y
        assert X_train.shape[0] == Y_train.shape[0]
        num_batches = int(np.ceil(X_train.shape[0] / batch))
        step = 0;
        steps = []
        X_batches = zip(np.array_split(X_train, num_batches, axis=0), np.array_split(Y_train, num_batches, axis=0))

        for ep in range(epochs):
            print("Epoch: ", ep, "===============================================")
            for x, y in X_batches:
                accuracy = 0
                predictions, loss ,weight_sum = LENET5.feedForward(x, y,self.layers)
                if len(y.shape) == 2:
                    temp_y = np.argmax(y, axis=1)
                    temp_predictions = np.argmax(predictions, axis=1)
                    accuracy = np.sum(np.equal(temp_y,temp_predictions))/len(temp_y)
                #loss = LENET5.loss_function(predictions, y, wsum=weight_sum, zeta=zeta)
                self.loss_history += [loss]
                LENET5.backpropagation(y, self.layers)          #check this gradient
                LENET5.update_parameters(self.layers, x.shape[0], alpha, zeta, method)
                print("Step: ", step, ":: Loss: ", loss, "weight_sum: ", weight_sum, "acc ::", accuracy)

            XY = list(zip(X_train, Y_train))
            np.random.shuffle(XY)
            new_X, new_Y = zip(*XY)
            assert len(new_X) == X_train.shape[0] and len(new_Y) == len(new_X)
            X_batches = zip(np.array_split(new_X, num_batches, axis=0), np.array_split(new_Y, num_batches, axis=0))
        pass



def main():
    X, Y = spiral.create_data(samples=150, classes=3)
    N = len(Y)
    Y_train = np.zeros((N, 3))
    Y_train[np.arange(N), Y[range(0, N)]] = 1
    #print(X.shape, Y_train.shape)
    lenet5 = LENET5(X, Y_train)
    lenet5.lenet_train()
if __name__ == '__main__':
    main()
    

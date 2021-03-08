# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:47:32 2021

@author: ASUS
"""
import os
from conv2 import *
from relu import *
from fc import *
from Activation_Softmax import Activation_Softmax
from loss import *
from maxpool import *
from Optimizer import *

import cv2
from einops import rearrange, reduce, repeat


import timeit
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from Datakain import Data


class LENET5:
    """docstring forLENET5."""
    def __init__(self, t_input, t_output, v_input, v_output, method, epochs):
        """
        Computes the forward pass of Conv Layer.
        Input:
            X: Input data of shape (N, D, H, W)
        Variables:
            kernel: Weights of shape (K, K_D, K_H, K_W)
            bias: Bias of each filter. (K)
        where, N = batch_size or number of images
               H, W = Height and Width of input layer
               D = Depth of input layer
               K = Number of filters/kernels or depth of this conv layer
               K_H, K_W = kernel height and Width

        Output:
        """
        # Conv Layer-1
        conv1 = CONV_LAYER((6, 48, 48), (6, 3, 5, 5), (2304, 13824), pad=2, stride=1,) #filename=b_dir+"conv0.npz")
        relu1 = RELU_LAYER()
        # Sub-sampling-1
        pool2 = MAX_POOL_LAYER(stride=2)
        # Conv Layer-2
        conv3 = CONV_LAYER((6, 24, 24), (6, 6, 3, 3), (576, 3456), pad=0, stride=1, )#filename=b_dir+"conv3.npz")
        relu3 = RELU_LAYER()
        # Sub-sampling-2
        pool4 = MAX_POOL_LAYER(stride=2)
        # Fully Connected-1
        fc5 = FC_LAYER(120, (726, 120), )#filename=b_dir+"fc6.npz")
        relu5 = RELU_LAYER()
        # Fully Connected-2
        fc6 = FC_LAYER(84, (120, 84), )#filename=b_dir+"fc8.npz")
        relu6 = RELU_LAYER()
        # Fully Connected-3
        output = FC_LAYER(4, (84, 4), )#filename=b_dir+"fc10.npz")
        softmax_crossentropy = Activation_Softmax_Loss_CategoricalCrossentropy()
        ##softmax = SOFTMAX_LAYER()
        self.layers = [conv1, relu1, pool2, conv3, relu3, pool4, fc5, relu5, fc6, relu6, output, softmax_crossentropy]

        self.X = t_input
        self.Y = t_output
        self.Xv = v_input
        self.Yv = v_output
        self.method = method
        self.epochs = epochs
        
        self.acc_history = []
        self.loss_history = []

        self.epoch_acc = []
        self.epoch_loss = []
        self.epoch_weight = []
        
        self.weights_history = []
        self.gradient_history = []
        
        self.valid_loss_history = []
        self.valid_acc_history = []
        self.valid_steps = []


    @staticmethod
    def one_image_time(X, layers):
        """
        Computes time of conv and fc layers
        Input:
            X: Input
            layers: List of layers.
        Output:
            inp: Final output
        """
        inp = X
        conv_time = 0.0
        fc_time = 0.0
        layer_time = []

        for layer in layers:
            start = timeit.default_timer()
            if isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            else:
                inp, ws = layer.forward(inp)
            stop = timeit.default_timer()
            layer_time += [stop-start]
            if isinstance(layer, (FC_LAYER, SIGMOID_LAYER, SOFTMAX_LAYER)):
                fc_time += stop - start
            if isinstance(layer, (CONV_LAYER, RELU_LAYER)):
                conv_time += stop - start
        return conv_time, fc_time, layer_time


    @staticmethod
    def feedForward(X, layers, y_true):
        """
        Computes final output of neural network by passing
        output of one layer to another.
        Input:
            X: Input
            layers: List of layers.
        Output:
            inp: Final output
        """
        inp = X
        wsum = 0
        y_true = y_true
        for layer in layers:
            if isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            elif isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                loss, ws = layer.forward(inp,y_true)
                inp = layer.output
            else:
                inp, ws = layer.forward(inp)
            wsum += ws
        return inp, loss, wsum

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
            else :
                delta = layer.backward(delta)

    @staticmethod
    def update_parameters(layers, optimizer ,batch_size, a, z, m):
        """
        Update weight parameters of each layer
        """
        for layer in layers:
            if isinstance(layer, (CONV_LAYER, FC_LAYER)):
                #layer.update_kernel(batch=batch_size, alpha=a, zeta=z, method=m)
                optimizer.update_params(layer)


    def lenet_train(self, **params):
        """
        Train the Lenet-5.
        Input:
            params: parameters including "batch", "alpha"(learning rate),
                    "zeta"(regularization parameter), "method" (gradient method),
                    "epochs", ...
        """
        batch  = params.get("batch", 32)             # Default 50
        alpha  = params.get("alpha", 0.01)            # Default 0.1
        zeta   = params.get("zeta", 0)               # Default 0 (No regularization)
        method = params.get("method", "adam")            # Default
        epochs = params.get("epochs", 500)             # Default 4
        print("Training on params: batch=", batch, " learning rate=", alpha, " L2 regularization=", zeta, " method=", method, " epochs=", epochs)
        
        print(method)
        X_train, Y_train = self.X, self.Y
        assert X_train.shape[0] == Y_train.shape[0]
        num_batches = int(np.ceil(X_train.shape[0] / batch))
        self.n_step, step = 0,0;
        steps = []
        X_batches = zip(np.array_split(X_train, num_batches, axis=0), np.array_split(Y_train, num_batches, axis=0))
        prev_loss = 0
        if not hasattr(self, 'optimizer'):
            if (method == "gd_momentum"):
                self.optimizer = Optimizer_SGD(learning_rate=alpha, decay=0.0, momentum=0.0)
            elif (method == "adam"):
                self.optimizer = Optimizer_Adam(learning_rate=alpha, decay=0.0,)
            elif (method =="adagrad"):
                    self.optimizer = Optimizer_Adagrad(learning_rate=alpha, decay=0., epsilon=1e-7)
            elif (method == "rmsprop"):
                    self.optimizer = Optimizer_RMSprop(learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9)
        
        for ep in range(epochs):
            temp_loss, temp_acc,temp_weight = [],[],[]
            loss = 0.0
            self.optimizer.pre_update_params()
            print("Epoch: ", ep, "===============================================")
            for x, y in X_batches:
                predictions, loss ,weight_sum = LENET5.feedForward(x, self.layers, y)
                if len(y.shape) == 2 and len(predictions.shape) == 2 :
                    temp_y = np.argmax(y, axis=1)
                    temp_predictions = np.argmax(predictions, axis=1)
                    accuracy = np.sum(np.equal(temp_y,temp_predictions))/len(temp_y)

                else :
                    accuracy = np.mean(predictions==y)
             
                temp_acc.append(accuracy)
                temp_loss.append(loss)
                temp_weight.append(weight_sum)

                LENET5.backpropagation(y, self.layers)          #check this gradient
                LENET5.update_parameters(self.layers, self.optimizer ,x.shape[0], alpha, zeta, method)
                print("Step::", step, "\tAcc::", accuracy,"\tLoss:: ", loss, "\tWeight_sum:: ", weight_sum)
                if step % 10 == 0:
                    pred, v_loss, w = LENET5.feedForward(self.Xv, self.layers, self.Yv)
                    #v_loss = LENET5.loss_function(pred, self.Yv, wsum=w, zeta=zeta)
                    print("Validation error: ", v_loss)
                    if len(pred.shape) == 2 and len(self.Yv.shape) == 2 :
                        temp_y = np.argmax(self.Yv, axis=1)
                        temp_predictions = np.argmax(pred, axis=1)
                        v_acc = np.sum(np.equal(temp_y,temp_predictions))/len(temp_y)
                        print("Validation acc: ", v_acc)
                    else :
                        v_acc = np.mean(pred==Yv)
                    self.valid_steps += [step]
                    self.valid_loss_history += [v_loss]
                    self.valid_acc_history += [v_acc]

                step += 1
            self.optimizer.post_update_params()
            
            self.weights_history += [temp_weight]
            self.loss_history += [temp_loss]
            self.acc_history += [temp_acc]
            
            average_weight = np.average(np.array(temp_weight))
            average_loss = np.average(np.array(temp_loss))
            average_acc = np.average(np.array(temp_acc))

            self.epoch_loss += [average_loss]
            self.epoch_acc += [average_acc]
            self.epoch_weight += [average_weight]
            
            print("Epoch::{} \tAve acc::{} \tAve loss::{} \tAve weight::{}".format(ep, average_acc,average_loss,average_weight))
            
            #if
            XY = list(zip(X_train, Y_train))
            np.random.shuffle(XY)
            new_X, new_Y = zip(*XY)
            assert len(new_X) == X_train.shape[0] and len(new_Y) == len(new_X)
            X_batches = zip(np.array_split(new_X, num_batches, axis=0), np.array_split(new_Y, num_batches, axis=0))
        
        self.n_steps=step
        print("Total Epoch::{} ============= Total Step::{}".format(self.epochs,self.n_step))

        pass

    def lenet_predictions(self, X, Y):
        """
        Predicts the ouput and computes the accuracy on the dataset provided.
        Input:
            X: Input of shape (Num, depth, height, width)
            Y: True output of shape (Num, Classes)
        """
        start = timeit.default_timer()
        predictions, loss ,weight_sum = LENET5.feedForward(X, self.layers, Y)
        stop = timeit.default_timer()

        #loss = LENET5.loss_function(predictions, Y, wsum=weight_sum, zeta=0.99)
        y_true = np.argmax(Y, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        print("Dataset accuracy: ", accuracy_score(y_true, y_pred)*100)
        print("FeedForward time:", stop - start)
        pass

    @staticmethod
    def plots(x, y, z, steps):
        try:
            plt.figure(1)
            plt.plot(x, '-bo', label="Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss', fontsize=18)
            plt.title('Training Error rate vs Number of iterations')
            plt.savefig("Loss_function_vs_iter.jpeg")
        except:
            pass

        try:
            plt.figure(2)
            plt.plot(steps, y, '-bo', label="Training Loss")
            plt.plot(steps, z, '-ro', label="Validation Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss Value', fontsize=18)
            plt.title('Training and Validation error rates vs number of iterations')
            plt.legend(loc='upper right')
            plt.savefig("error_rates.jpeg")
        except:
            pass
        pass
    
    def save_parameters(self,mainPath):
        """
        Saves the weights and biases of Conv and Fc layers in a file.
        """
        path = os.path.join(mainPath,self.method)
        if not os.path.exists(path):
            os.makedirs(path)
        for layer in self.layers:
            if isinstance(layer, CONV_LAYER):
                if not hasattr(layer, 'layer_name'):
                    layer_name = "conv" + str(self.layers.index(layer))
                    layer.layer_name = layer_name
                np.savez(str(path) + "/" + layer.layer_name + "-" + str(self.method) + str(self.epochs), layer.kernel, layer.bias)
            
            elif isinstance(layer, FC_LAYER):
                if not hasattr(layer, 'layer_name'):
                    layer_name = "fc" + str(self.layers.index(layer))
                    layer.layer_name = layer_name
                np.savez(str(path) + "/" + layer.layer_name + "-" + str(self.method) + str(self.epochs), layer.kernel, layer.bias)
        
        np.savez(str(path) + "/" + "valid_history" +"-" + str(self.method) + str(self.epochs), self.valid_loss_history, self.valid_acc_history,self.valid_steps )
        np.savez(str(path) + "/" + "step_history" +"-" + str(self.method) + str(self.epochs), self.loss_history, self.acc_history,self.weights_history, self.n_steps )
        np.savez(str(path) + "/" + "epoch_history" +"-" + str(self.method) + str(self.epochs), self.epoch_loss, self.epoch_acc, self.epoch_weight,self.epochs )
        
        pass
    
    def load_train_details(self, **params):
        method = params.get("method", self.method)
        epochs = params.get("epochs", self.epochs)
        mainPath = params.get("mainPath", "")
        
        fname = "valid_history" +"-" + str(method) + str(epochs) + ".npz"
        path = os.path.join(mainPath,str(method),fname)
        arr_files = np.load(path)
        self.valid_loss_history, self.valid_acc_history,self.valid_steps = arr_files['arr_0'], arr_files['arr_1'], arr_files['arr_2']
        
        fname = "step_history" +"-" + str(method) + str(epochs) + ".npz"
        path = os.path.join(mainPath,str(method),fname)
        arr_files = np.load(path)
        self.loss_history, self.acc_history,self.weights_history, self.n_steps = arr_files['arr_0'], arr_files['arr_1'], arr_files['arr_2'], arr_files['arr_3']
        
        fname = "epoch_history" +"-" + str(method) + str(epochs) + ".npz"
        path = os.path.join(mainPath,str(method),fname)
        arr_files = np.load(path)
        self.epoch_loss, self.epoch_acc, self.epoch_weight,self.epochs = arr_files['arr_0'], arr_files['arr_1'], arr_files['arr_2'], arr_files['arr_3']
        """
        for i in range(0,epochs,1):
            print("Epoch:: {} Loss::{} \t\tAcc::{} \tM-Weight::{}".format(i, self.epoch_loss[i], self.epoch_acc[i], self.epoch_weight[i]))
        for i in range(0,len(self.valid_steps),1):
            print("Val-Step:: {} Loss::{} \t\tAcc::{}".format(self.valid_steps[i], self.valid_loss_history[i], self.valid_acc_history[i]))
        print(self.loss_history.shape, self.acc_history.shape,self.weights_history.shape, self.n_steps)
        for i in range(0,self.n_steps,1):
            print("Steps:: {} Loss::{} \t\tAcc::{} \tM-Weight::{}".format(i, self.loss_history[i], self.acc_history[i], self.weights_history[i] ))
        """
    def load_parameters(self, **params):
        method = params.get("method", "adam")
        epochs = params.get("epochs", 500)
        mainPath = params.get("mainPath", "")
        tqdm._instances.clear()
        for layer in tqdm(self.layers, desc="loading layers..."):
            if isinstance(layer, CONV_LAYER):
                if not hasattr(layer, 'layer_name'):
                    layer_name = "conv" + str(self.layers.index(layer))
                    layer.layer_name = layer_name
                #fname = "conv" + str(self.layers.index(layer)) + "-" + str(self.method) + ".npz"
                fname = layer.layer_name + "-" + str(method) + str(epochs) + ".npz"
                path = os.path.join(mainPath,str(method),fname)
                layer.load(path,layer.kernel.shape)
                
            elif isinstance(layer, FC_LAYER):
                if not hasattr(layer, 'layer_name'):
                    layer_name = "fc" + str(self.layers.index(layer))
                    layer.layer_name = layer_name
                fname = layer.layer_name + "-" + str(method) + str(epochs) +".npz"
                path = os.path.join(mainPath,str(method),fname)
                layer.load(path,layer.kernel.shape)
                

    def check_gradient(self):
        """
        Computes the numerical gradient and compares with Analytical gradient
        """
        sample = 10
        epsilon = 1e-4
        X_sample = self.X[range(sample)]
        Y_sample = self.Y[range(sample)]
        predictions, weight_sum = LENET5.feedForward(X_sample, self.layers)
        LENET5.backpropagation(Y_sample, self.layers)

        abs_diff = 0
        abs_sum = 0

        for layer in self.layers:
            if not isinstance(layer, (CONV_LAYER, FC_LAYER)):
                continue
            i = 0
            print("\n\n\n\n\n")
            print(type(layer))
            del_k = layer.delta_K + (0.99*layer.kernel/sample)
            kb = chain(np.nditer(layer.kernel, op_flags=['readwrite']), np.nditer(layer.bias, op_flags=['readwrite']))
            del_kb = chain(np.nditer(del_k, op_flags=['readonly']), np.nditer(layer.delta_b, op_flags=['readonly']))

            for w, dw in zip(kb, del_kb):
                w += epsilon
                pred, w_sum = LENET5.feedForward(X_sample, self.layers)
                loss_plus = LENET5.loss_function(pred, Y_sample, wsum=w_sum, zeta=0.99)

                w -= 2*epsilon
                pred, w_sum = LENET5.feedForward(X_sample, self.layers)
                loss_minus = LENET5.loss_function(pred, Y_sample, wsum=w_sum, zeta=0.99)

                w += epsilon
                numerical_gradient = (loss_plus - loss_minus)/(2*epsilon)

                abs_diff += np.square(numerical_gradient - dw)
                abs_sum  += np.square(numerical_gradient + dw)
                print(i, "Numerical Gradient: ", numerical_gradient, "Analytical Gradient: ", dw)
                if not np.isclose(numerical_gradient, dw, atol=1e-4):
                    print("Not so close")
                if i >= 10:
                    break
                i += 1

        print("Relative difference: ", np.sqrt(abs_diff)/np.sqrt(abs_sum))
        pass
    
    @staticmethod
    def one_image(layers, path):
        
        
        inp = cv2.imread(path)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = rearrange(inp, ' h w c ->  c h w ')
        for layer in layers:
            if isinstance(layer,CONV_LAYER) and len(inp.shape) == 3:
                inp, ws = layer.forward(inp.reshape(1,inp.shape[0],inp.shape[1],inp.shape[2]))
            elif isinstance(layer, FC_LAYER) and len(inp.shape) == 4:
                inp, ws = layer.forward(inp.reshape(inp.shape[0], inp.shape[1]*inp.shape[2]*inp.shape[3]))
            elif isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                output = layer.guessing(inp)
            else:
                inp, ws = layer.forward(inp)
        #print("prob",inp)
        return output

    
def main():
    mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
    workPath = os.path.split(mainPath) #path working folder (whole file project)
    imagePath = "data_kain"
    
    layer_time = []
    data = Data(workPath, imagePath)
    X_train, trainLabel, X_test, testLabel = data.load()
    kelas = data.jum_kelas
    len_label = trainLabel.shape[0]
    
    Y_train = np.zeros((len_label,kelas))
    Y_train[np.arange(len_label), trainLabel[range(0, len_label)]] = 1
    
    kelas = data.jum_kelas
    len_label = testLabel.shape[0]
    Y_test = np.zeros((len_label, kelas))
    Y_test[np.arange(len_label), testLabel[range(0, len_label)]] = 1
    
    method = "adam"
    epochs = 500
    mylenet = LENET5(X_train, Y_train, X_test, Y_test, method=method,epochs=epochs)
    
    """ Train """
    """
    start = timeit.default_timer()
    mylenet.lenet_train(method=method, epochs=epochs, batch=32, alpha=0.01, zeta=0)
    stop = timeit.default_timer()
    print("Training time:", stop - start)
    print("Training ", end="")
    
    mylenet.save_parameters(mainPath)
    mylenet.lenet_predictions(X_test, Y_test)
    """
    mylenet.load_parameters(mainPath=mainPath,epochs=epochs,method=method,)
    mylenet.lenet_predictions(X_test, Y_test)
    
    """ load training history
    mylenet.load_train_details(mainPath=mainPath,epochs=epochs,method=method,)
    
    """
    """ testing one image """
    
    imgpath= "C:/Users/ASUS/Documents/py/cnn-numpy/data_kain/gringsing/gringsing_61.jpg"
    temp = os.path.split(imgpath)
    prob = mylenet.one_image(mylenet.layers, imgpath )
    print("image test::",imgpath)
    print("\nFile Name ::", temp[1], " Tipe kain ::", data.labelName[np.argmax(prob)], "||" ,
          "confidence ::", prob[0,np.argmax(prob)])
    
if __name__=='__main__':
    main()

    


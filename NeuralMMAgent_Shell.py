# Some potentially useful modules
import random
import numpy
import math
import matplotlib.pyplot as plt
import time
import copy


class NeuralMMAgent(object):
    '''
    Class to for Neural Net Agents that compete in the Mob task
    '''
    
    def __init__(self, num_in_nodes, num_hid_nodes, num_hid_layers, num_out_nodes, learning_rate = 0.2, max_epoch=10000, min_sse=.01, momentum=0, creation_function=None, activation_function=None, random_seed=1):
        '''
        Arguments:
            num_in_nodes -- total # of input nodes for Neural Net
            num_hid_nodes -- total # of hidden nodes for each hidden layer in the Neural Net
            num_hid_layers -- total # of hidden layers for Neural Net
            num_out_nodes -- total # of output layers for Neural Net
            learning_rate -- learning rate to be used when propogating error
            max_epoch -- maximum number of epochs for our NN to run during learning
            min_sse -- minimum SSE that we will use as a stopping point
            momentum -- Momentum term used for learning
            creation_function -- function that will be used to create the
                neural network given the input
            activation_function -- list of two functions:
                1st function will be used by network to determine activation given a weighted summed input
                2nd function will be the derivative of the 1st function
            random_seed -- used to seed object random attribute.
                This ensures that we can reproduce results if wanted
        '''
        assert num_in_nodes > 0 and num_hid_layers > 0 and num_hid_nodes and num_out_nodes > 0, "Illegal number of input, hidden, or output layers!"
        self.num_in_nodes=num_in_nodes
        self.num_hid_nodes=num_hid_nodes
        self.num_hid_layers=num_hid_layers
        self.num_out_nodes=num_out_nodes
        self.learning_rate=learning_rate
        self.max_epoch=max_epoch
        self.min_sse=min_sse
        self.momentum=momentum


        input_list1=[0.0]*num_in_nodes
        input_list2=[[0.0]*num_hid_nodes]*num_hid_layers
        input_list3=[0.0]*num_out_nodes
        combined_active_list=[input_list1]
        combined_active_list.extend(input_list2)
        combined_active_list.append(input_list3)
        self.sample=combined_active_list
        self.little_delta=combined_active_list



        if(num_hid_layers==1):
            #print(num2)
            input_list2=[0.0]*num_hid_nodes
            combined_active_list=[input_list2]
        elif(num_hid_layers>1):
            input_list2=[[0.0]*num_hid_nodes]*num_hid_layers
            combined_active_list.extend(input_list2)
        input_list3=[0.0]*num_out_nodes
        combined_active_list.append(input_list3)
        
        self.bias=combined_active_list
        print(self.bias)
        self.little_bias=combined_active_list


        num1=num_in_nodes*num_hid_nodes
        num2=num_hid_nodes*num_hid_nodes
        num3=num_out_nodes*num_hid_nodes
        num_hid_layers-1
        input_list1=[0.0]*num1
        combined_active_list1=[input_list1]
        if(num_hid_layers==1):
            #print(num2)
            input_list2=[[0.0]*num3]
            combined_active_list1.append(input_list2)
        elif(num_hid_layers>1):
            input_list2=[[0.0]*num2]*num_hid_layers-1
            combined_active_list1.extend(input_list2)
            input_list3=[0.0]*num3
            combined_active_list1.append(input_list3)

        
        self.weights=combined_active_list1
        self.little_weights=combined_active_list1
        #print(self.weights)

        for i in range(len(self.weights)):
            #print(i)
            for j in range(len(self.weights[i])):
                self.weights[i][j]=random.uniform(-0.5,0.5)
        self.num_wire=[num_in_nodes*num_hid_nodes,num_hid_nodes*num_hid_nodes,num_hid_nodes*num_out_nodes]
        pass



        
    



    def train_net_incremental(self, input_list, output_list, max_num_epoch=100000, min_sse=0.001):
        ''' Trains neural net using incremental learning
            (update once per input-output pair)
            Arguments:
                input_list -- 2D list of inputs
                output_list -- 2D list of outputs matching inputs
            Outputs:
                1d list of errors (total error each epoch) (e.g., [0.1])
        '''
        all_err = []
        for epoch in range(max_num_epoch):
            total_err = 0
            for row in range(len(input_list)):
                # Feed forward
                activations = self._feed_forward(input_list, row)

                # Calculate error for the output nodes
                errors = []
                for i in range(self.num_out_nodes):
                    errors.append([output_list[row][i] - activations[-1][i]])
                    total_err += errors[-1][0] ** 2

                # Calculate deltas
                little_deltas,weight_deltas, bias_deltas = self._calculate_deltas(activations, errors)
                # Adjust weights and biases
                weight_deltas, bias_deltas = self._adjust_weights_bias(weight_deltas, bias_deltas)
                self.weights = weight_deltas
                self.bias = bias_deltas
            total_err=total_err/2
            all_err.append(total_err)
            if (total_err < min_sse):
                break
        return all_err







    def _feed_forward(self, input_list, row):
        '''
        Used to feedforward input and calculate all activation values
            Arguments:
                input_list -- a list of possible input values
                row -- the row from that input_list that we should use
            Outputs:
                list of activation values
        '''

        output_list=self.sample
        #print(output_list)
        for i in range(self.num_hid_layers+2):          
            if i==0:
                for j in range(len(input_list[i])):
                    output_list[i][j]=input_list[row][j]
            elif i==self.num_hid_layers+1:
                for j in range(self.num_out_nodes-1):
                    nodeValue=0
                    for k in range(self.num_hid_nodes):
                        nodeValue+=output_list[i-1][k]*self.weights[0][k*self.num_hid_nodes+j]
                    output_list[i][j]=self.sigmoid_af(nodeValue+self.bias[i][j])#check
            else:
                for j in range(self.num_hid_nodes-1):
                    if i==1:
                        nodeValue=0
                        
                        for k in range(len(output_list[0])):
                            #print(j,k)
                            #print(self.weights[0][k*self.num_hid_nodes+j])
                            #print(self.weights)
                            nodeValue+=output_list[0][k]*self.weights[0][k*self.num_hid_nodes+j]
                        print(i,j)
                        print(self.bias)
                        print(self.bias[i][j])
                        print(output_list[i][j])
                        output_list[i][j]=self.sigmoid_af(nodeValue+self.bias[i][j])
                    else:
                        nodeValue=0
                        for k in range(self.num_hid_nodes):
                            nodeValue+=output_list[i-1][k]*self.weights[0][k*self.num_hid_nodes+j]
                        output_list[i][j]=self.sigmoid_af(nodeValue+self.bias[i][j])
        return output_list
        
                    
    

    def _calculate_deltas(self, activations, errors, prev_weight_deltas=None):
        '''
        Used to calculate all weight deltas for our neural net
            Parameters:
                activations -- a 2d list of activation values
                errors -- a 2d list of errors
                prev_weight_deltas [OPTIONAL] -- a 2d list of previous weight deltas
            Output:
                A tuple made up of 3 items:
                    A 2d list of little deltas (e.g., [[0, 0], [-0.1, 0.1], [0.1]])
                    A 2d list of weight deltas (e.g., [[-0.1, 0.1, -0.1, 0.1], [0.1, 0.1]])
                    A 2d list of bias deltas (e.g., [[0, 0], [-0.1, 0.1], [0]])
        '''
        #double check

        little_deltas=self.little_delta
        weight_deltas=self.little_weights
        bias_deltas=self.little_bias
        for i in range(self.num_hid_layers+2,-1,-1):
            if i==0:
                for j in range(self.num_in_nodes):
                    for k in range(self.num_hid_nodes):
                        nodeValue+=little_deltas[i+1][k]*self.weights[0][k*self.num_hid_nodes+j]*self.sigmoid_af_deriv(activations[1][k])
                    little_deltas[i][j]=nodeValue
            elif i==self.num_hid_layers+2:
                for j in range(self.num_out_nodes):
                    nodeValue=errors[-1][j]
                    little_deltas[i-1][j]=nodeValue #check
            elif i==self.num_hid_layers+1:
                    nodeValue=0
                    for k in range(self.num_out_nodes):
                        nodeValue+=little_deltas[i][k]*self.weights[i-1][k*self.num_hid_nodes+j]*self.sigmoid_af_deriv(activations[i][k])
                    little_deltas[i-1][j]=nodeValue
            else:
                for j in range(self.num_hid_nodes):
                    nodeValue=0
                    for k in range(self.num_hid_nodes):
                        nodeValue+=little_deltas[i][k]*self.weights[i-1][k*self.num_hid_nodes+j]*self.sigmoid_af_deriv(activations[i][k])
                    little_deltas[i-1][j]=nodeValue
        little_deltas_save=little_deltas
        #print(little_deltas)
        for i in range(self.num_hid_layers+1,0,-1):
            if i==0:
                count=0
                for j in range(self.num_in_nodes):
                    for k in range(self.num_hid_nodes):
                        weight_deltas[i][count]=little_deltas[i+1][j]*self.sigmoid_af_deriv(activations[i+1][k])*activations[i][j]
                        count=count+1


            elif i>0 and i<self.num_hid_layers:
                count=0
                for j in range(self.num_hid_nodes):
                    for k in range(self.num_hid_nodes):
                        weight_deltas[i][count]=little_deltas[i+1][k]*self.sigmoid_af_deriv(activations[i+1][k])*activations[i][j]
                        count=count+1


            elif i==self.num_hid_layers:
                count=0
                for j in range(self.num_hid_nodes-1):
                    for k in range(self.num_out_nodes):
                        #print(i,count,j,k)
                        #print(little_deltas[i+1][k])
                        #print(activations[i+1][k])
                        #print(activations[i][j])
                        #print(weight_deltas[i][count])
                        weight_deltas[i][count]=little_deltas[i+1][j]*self.sigmoid_af_deriv(activations[i+1][k])*activations[i][j]
                        count=count+1

        #print(activations)
        #print(bias_deltas)
        for i in range(self.num_hid_layers+1,0,-1):
            if i==0:
                #print(little_deltas_save)
                break
            else:
                for j in range(len(activations[i])-1):
                    #print(i,j)
                    #print(bias_deltas[i][j])
                    #print(little_deltas[i][j])
                    #print(activations[i][j])
                    #print(bias_deltas[i][j])
                    bias_deltas[i][j]=little_deltas[i][j]*self.sigmoid_af_deriv(activations[i][j])
        return little_deltas_save,weight_deltas,bias_deltas
    

        #Calculate error gradient for each output node & propgate error
        #   (calculate weight deltas going backward from output_nodes)




    def _adjust_weights_bias(self, weight_deltas, bias_deltas):
        '''
        Used to apply deltas
        Parameters:
            weight_deltas -- 2d list of weight deltas
            bias_deltas -- 2d list of bias deltas
        Outputs:
            A tuple w/ the following items (in order):
            2d list of all weights after updating (e.g. [[-0.071, 0.078, 0.313, 0.323], [-0.34, 0.021]])
            list of all biases after updating (e.g., [[0, 0], [0, 0], [0]])
        '''
        #previous weights"self.weights" - learning rate *weight_deltas
        #previous bias "self.bias" - learning rate * weight_deltas
        for i in range(len(weight_deltas)):
            for j in range(len(weight_deltas[i])):
                weight_deltas[i][j]=weight_deltas[i][j]*self.learning_rate
        for i in range(len(bias_deltas)):
            for j in range(len(bias_deltas[i])):
                #print(i,j)
                #print(bias_deltas)
                #print(bias_deltas[i][j])
                bias_deltas[i][j]=bias_deltas[i][j]*self.learning_rate
        return weight_deltas, bias_deltas       


    #########ACCESSORS

    def get_weights(self):
        return (self.weights)

    def set_weights(self, weights):
        self.weights = weights

    def get_biases(self):
        return (self.bias)

    def set_biases(self, bias):
        self.bias = bias

    ################

    @staticmethod
    def sigmoid_af(summed_input):
        #Sigmoid function
        return 1/(1+math.exp(-summed_input))

    @staticmethod
    def sigmoid_af_deriv(sig_output):
        #the derivative of the sigmoid function
        return sig_output*(1-sig_output)


#----#
#Some quick test code

test_agent = NeuralMMAgent(2, 2, 1, 1,random_seed=5, max_epoch=1000000, learning_rate=0.2, momentum=0)
test_in = [[1,0],[0,0],[1,1],[0,1]]
test_out = [[1],[0],[0],[1]]
test_agent.set_weights([[-.37,.26,.1,-.24],[-.01,-.05]])
test_agent.set_biases([[0,0],[0,0],[0,0]])
all_errors = test_agent.train_net_incremental(test_in, test_out, max_num_epoch = test_agent.max_epoch, min_sse = test_agent.min_sse)


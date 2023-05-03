"""=================================================== Assignment 4 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc. 

"""


""" Import the required libraries"""
# Start your code here
import numpy as np
import matplotlib.pyplot as plt

# End your code here


def func(x_input):
    """
    --------------------------------------------------------
    Write your logic to evaluate the function value. 

    Input parameters:
        x: input column vector (a numpy array of n dimension)

    Returns:
        y : Value of the function given in the problem at x.
        
    --------------------------------------------------------
    """
    
    # Start your code here
    x1 = x_input.T[0][0]
    x2 = x_input.T[0][1]
    y = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
    # End your code here
    
    return y

def gradient(func, x_input):
    """
    --------------------------------------------------------------------------------------------------
    Write your logic for gradient computation in this function. Use the code from assignment 2.

    Input parameters:  
      func : function to be evaluated
      x_input: input column vector (numpy array of n dimension)

    Returns: 
      delF : gradient as a column vector (numpy array)
    --------------------------------------------------------------------------------------------------
    """
    # Start your code here
    # Use the code from assignment 2
    h = 0.001
    dim_x = x_input.shape[0]
    x_input = x_input.reshape((dim_x, 1))
    delF = np.zeros((dim_x, 1))
    I_mat = np.identity(dim_x)
    for i in range(dim_x):
        I_vec = I_mat[i, :].reshape((dim_x, 1))
        delF[i] = (func(x_input + h*I_vec) - func(x_input - h*I_vec))/(2*h)
    # End your code here
    return delF 
        
def FRCG(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for FR-CG using in-exact line search. 

    Input parameters:  
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector(numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """
    
    # Start your code here
    dim_X = x_initial.shape[0] 
    N = 15000
    epsilon = 1e-6
    k = 1
    X0 = x_initial
    X1 = x_initial
    alpha_bar = 5
    rho = 0.8
    c = 0.1
    alpha_k = 0
    grad_f = gradient(func,X0)
    p_k = -1*grad_f
    f_values = [func(X0)]
    X = x_initial.reshape((dim_X,1))

    while grad_f.T@grad_f > epsilon and k <= N:
        alpha = alpha_bar
        while func(X0+alpha*p_k) > func(X0) + c*alpha*(grad_f.T)@p_k:
            alpha = rho*alpha
        alpha_k = alpha

        X1 = X0 + alpha_k*p_k

        grad_f1 = gradient(func,X1)
        beta_k1 = grad_f1.T@grad_f1/(grad_f.T@grad_f)
        p_k = -1*grad_f1 + beta_k1*p_k

        X = np.concatenate((X,X1),axis=1)
        f_values.append(func(X1))
        k+= 1
        grad_f = grad_f1
        X0 = X1
    
    x_output = X1
    f_output = func(X1)
    grad_output = gradient(func,X1)
    if k == 15000 and grad_output >= epsilon:
        print('Maximum iterations reached but convergence did not happen')
    print("X =",x_output,"f(X) =",f_output,"gradF(X) =",grad_output)
    plot_x_iterations(k,X.T,"FR-CG Method")
    plot_func_iterations(k,f_values,"FR-CG Method")
    # End your code here
    
    return x_output, f_output, grad_output
    
def plot_x_iterations(NM_iter, NM_x, method_name):
  """
  -----------------------------------------------------------------------------------------------------------------------------
  Write your logic for plotting x_input versus iteration number i.e,
  x1 with iteration number and x2 with iteration number in same figure but as separate subplots. 

  Input parameters:  
    NM_iter : no. of iterations taken to converge (integer)
    NM_x: values of x at each iterations, a (num_interations X n) numpy array where, n is the dimension of x_input

  Output the plot.
  -----------------------------------------------------------------------------------------------------------------------------
  """
  # Start your code here
  dim_X = NM_x.shape[1]
  itr_vec = [i+1 for i in range(NM_iter)]
  itr_vec = np.array(itr_vec)
  
  for i in range(dim_X):
    x_vec = NM_x[:,i]
    plt.subplot(dim_X,1,i+1)
    if i == 0:
        plt.title(method_name)
    plt.xlabel("iterations")
    plt.ylabel("X"+str(i+1))
    plt.plot(itr_vec,x_vec)
  plt.show()
  # End your code here

def plot_func_iterations(NM_iter, NM_f, method_name):
  """
  ------------------------------------------------------------------------------------------------
  Write your logic to generate a plot which shows the value of f(x) versus iteration number.

  Input parameters:  
    NM_iter : no. of iterations taken to converge (integer)
    NM_f: function values at each iteration (numpy array of size (num_iterations x 1))

  Output the plot.
  -------------------------------------------------------------------------------------------------
  """
  # Start your code here
  itr_vec = [i+1 for i in range(NM_iter)]
  itr_vec = np.array(itr_vec)
  plt.title(method_name)
  plt.xlabel("iterations")
  plt.ylabel("func value")
  plt.plot(itr_vec,NM_f)
  plt.show()
  # End your code here

    
"""--------------- Main code: Below code is used to test the correctness of your code ---------------

    func : function to evaluate the function value. 
    x_initial: initial value of x, a column vector, numpy array
    
"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output = FRCG(func, x_initial)


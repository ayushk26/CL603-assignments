"""=================================================== Assignment 6 ===================================================

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
    y = x1**2 + x2**2 + (0.5*x1 + x2)**2 + (0.5*x1 + x2)**4
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


def hessian(func, x_input):
  """
  --------------------------------------------------------------------------------------------------
  Write your logic for hessian computation in this function. Use the code from assignment 2.

  Input parameters:  
    func : function to be evaluated
    x_input: input column vector (numpy array)

  Returns: 
    del2F : hessian as a 2-D numpy array
  --------------------------------------------------------------------------------------------------
  """
  # Start your code here
  # Use the code from assignment 2
  h = 0.001
  dim_x = x_input.shape[0]
  x_input = x_input.reshape((dim_x,1))
  del2F = np.zeros((dim_x,dim_x))
  I_mat = np.identity(dim_x)
  for i in range(dim_x):
    for j in range(dim_x):
      I_vec_i = I_mat[i,:].reshape((dim_x,1))
      I_vec_j = I_mat[j,:].reshape((dim_x,1))
      if i == j:
        del2F[i][j] = (func(x_input + h*I_vec_i) + func(x_input - h*I_vec_i) - 2*func(x_input))/(h)**2
      else:
        A = func(x_input + h*I_vec_i + h*I_vec_j)
        B = func(x_input - h*I_vec_i - h*I_vec_j)
        C = func(x_input - h*I_vec_i + h*I_vec_j)
        D = func(x_input + h*I_vec_i - h*I_vec_j)
        del2F[i][j] = (A+B-C-D)/(4*h*h)
  # End your code here
  
  return del2F

def model_func(func, x_val, p):
    m = func(x_val) + gradient(func,x_val).T@p + (1/2)*p.T@hessian(func,x_val)@p    
    return m

def get_p_newton(func,x_val):
    return -1*np.linalg.inv(hessian(func,x_val))@gradient(func,x_val)

def get_p_cauchy(func, x_val, delta):
    grad = gradient(func, x_val)
    grad_B_grad = grad.T@hessian(func, x_val)@grad
    
    if grad_B_grad > 0:  
        return -1*((grad.T@grad)/(grad_B_grad))*grad
    else:
        return -1*(delta/np.linalg.norm(grad))*grad

def TRPD(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for Trust Region - Powell Dogleg Method.

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
    delta_0 = 0.5
    delta_1 = 0
    delta_bar = 1
    n = 0.2
    k = 1
    gradF = gradient(func,X0)
    p_null = np.zeros(dim_X).reshape((dim_X,1))
    f_values = [func(X0)]
    X = x_initial.reshape((dim_X,1))
    

    while k<= N and gradF.T@gradF > epsilon:
        
        p_newton = get_p_newton(func,X0)
        p_cauchy = get_p_cauchy(func, X0, delta_0)
        if np.linalg.norm(p_newton) <= delta_0:
            pk = p_newton
        elif np.linalg.norm(p_cauchy) >= delta_0:
            pk = (delta_0/np.linalg.norm(p_cauchy))*p_cauchy
        else:
            A = (p_newton - p_cauchy).T@p_cauchy
            B = (np.linalg.norm(p_newton-p_cauchy))**2
            theta = (-A + (A**2 + (delta_0**2 - np.linalg.norm(p_cauchy)**2)*B)**0.5)/(B)
            pk = theta*p_newton + (1-theta)*p_cauchy
        rho_k = (func(X0) - func(X0 +pk))/(model_func(func,X0,p_null) - model_func(func,X0,pk))
        norm_pk = np.linalg.norm(pk)
        if rho_k < (1/4):
            delta_1 = (1/4)*norm_pk
        else:
            if rho_k > (3/4) and abs(norm_pk -delta_0) <epsilon:
                delta_1 = min(2*delta_0,delta_bar)
            else:
                delta_1 = delta_0
        
        if rho_k > n:
            X1 = X0 + pk
        else:
            X1 = X0
        X = np.concatenate((X,X1),axis=1)
        f_values.append(func(X1))
        X0 = X1
        delta_0 = delta_1
        gradF = gradient(func,X0)
        k+=1
    x_output = X1
    f_output = func(X1)
    grad_output = gradient(func, X1)
    
    if k == N and grad_output.T@grad_output > epsilon:
        print('Maximum iterations reached but convergence did not happen')
    plot_x_iterations(k,X.T,"TRPD Method")
    plot_func_iterations(k,f_values,"TRPD Method")
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

x_output, f_output, grad_output = TRPD(func, x_initial)

print("\n\nTrust Region - Powell Dogleg Method:")
print("-"*40)
print("\nFunction converged at x = \n",x_output)
print("\nFunction value at converged point = \n",f_output)
print("\nGradient value at converged point = \n",grad_output)
"""============================================ Assignment 3: Newton Method ============================================"""

""" Import the required libraries"""
# Start you code here
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
  x_input = x_input.reshape((dim_x,1))
  delF = np.zeros((dim_x,1))
  I_mat = np.identity(dim_x)
  for i in range(dim_x):
    I_vec = I_mat[i,:].reshape((dim_x,1))
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

def newton_method(func, x_initial):
  """
  -----------------------------------------------------------------------------------------------------------------------------
  Write your logic for newton method in this function. 

  Input parameters:  
    func : input function to be evaluated
    x_initial: initial value of x, a column vector (numpy array)

  Returns:
    x_output : converged x value, a column vector (numpy array)
    f_output : value of f at x_output
    grad_output : value of gradient at x_output
    num_iterations : no. of iterations taken to converge (integer)
    x_iterations : values of x at each iterations, a (num_interations x n) numpy array where, n is the dimension of x_input
    f_values : function values at each iteration (numpy array of size (num_iterations x 1))
  -----------------------------------------------------------------------------------------------------------------------------
  """
  # Write code here
  dim_X = x_initial.shape[0]
  X = x_initial.reshape((dim_X,1))
  X0 = x_initial.reshape((dim_X,1))
  X1 = np.zeros((dim_X,1))
  itr_count = 1
  epsilon = 10**(-6)
  f_values = [func(X0)]
  while itr_count <15000 and np.linalg.norm(gradient(func,X0)) >= epsilon:
    X1 = X0 - np.dot(np.linalg.inv(hessian(func,X0)),gradient(func,X0))
    X = np.concatenate((X,X1),axis=1)
    f_values.append(func(X1))
    X0 = X1
    itr_count += 1
  x_output = X1
  f_output = func(x_output)
  grad_output  = gradient(func,x_output)
  num_iterations = itr_count
  x_iterations = X.T
  f_values = np.array(f_values).reshape((itr_count,1))
  if itr_count == 15000 and np.linalg.norm(gradient(func,X0)) >= epsilon:
    print('Maximum iterations reached but convergence did not happen')
  # End your code here
  

  return x_output, f_output, grad_output, num_iterations, x_iterations, f_values


def plot_x_iterations(NM_iter, NM_x):
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
    plt.xlabel("iterations")
    plt.ylabel("X"+str(i+1))
    plt.plot(itr_vec,x_vec)
  plt.show()
  # End your code here

def plot_func_iterations(NM_iter, NM_f):
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
  plt.xlabel("iterations")
  plt.ylabel("func value")
  plt.plot(itr_vec,NM_f)
  plt.show()
  # End your code here



"""--------------- Main code: Below code is used to test the correctness of your code ---------------"""

x_initial = np.array([[1.5, 1.5]]).T

x_output, f_output, grad_output, num_iterations, x_iterations, f_values = newton_method(func, x_initial)

print("\nFunction converged at x = \n",x_output)
print("\nFunction value at converged point = \n",f_output)
print("\nGradient value at converged point = \n",grad_output)

plot_x_iterations(num_iterations, x_iterations)

plot_func_iterations(num_iterations, f_values)
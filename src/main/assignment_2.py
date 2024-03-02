import numpy as np

#Q1
def Neville_Method(x_points, y_points, x):
  
  matrix = np.zeros((len(x_points), len(x_points)))
  
  for counter, row in enumerate(matrix):
    row[0] = y_points[counter]
  
  numpoints = len(x_points)

  # Neville's formula inputted into the matrix
  for i in range(1, numpoints):
    for j in range(1, i+1):
      
      first_term = (x- x_points[i-j]) * matrix[i][j-1]
      second_term = (x - x_points[i]) * matrix[i-1][j-1]

      neville = (x_points[i] - x_points[i-j])

      num = (first_term - second_term)/neville

      matrix[i][j] = num
    
  return matrix[numpoints - 1][numpoints - 1] 

#Q2
def Newtons_Forward(x_points, y_points):
  lim = len(x_points)
  matrix = np.zeros((lim, lim))
  result = []

  for i in range(lim):
     matrix[i][0] = y_points[i]

  for i in range(1, lim):
    for j in range(1, i + 1):
      matrix[i][j] = (matrix[i][j - 1] - matrix[i - 1][j - 1]) / (x_points[i] - x_points[i - j])

      if i == j:
        result.append(matrix[i][j])
  
  print(result)
  return matrix

#Q3 
def Approximation_Method(matrix, x_points, value, start):
  
  x_index = 1.0
  result = start    
 
  for index in range(1, len(matrix)):
    coefficients = matrix[index][index]

    #We use Q2 result of x_points
    x_index *= (value - x_points[index - 1])
        
    
    f_x = coefficients * x_index

    #We use the result from the calculation of f_x
    result += f_x

  print(result)


#Q4
def Divided_Difference(matrix: np.array):
  dimensions_matrix = len(matrix)

  for i in range(0, dimensions_matrix):

    for j in range(0, i+0):
      
            
      # Input first cell column
      first_column: float = matrix[i-1][j]

      #Input diagonal of first cell column
      diag_first_column: float = matrix[i-1][j-1]

      #Calculate numerator by subtracting the diagonal of first column-first column
      n: float = diag_first_column - first_column
      
      d = matrix[i][j] - matrix[i][i-j]
    
      result= n / d
      matrix[i][j] = result
    
  return matrix

def Hermite_Approximation(x_points, y_points, f_x):
 
  dimensions_matrix = 2 * len(x_points)
  matrix = np.zeros((dimensions_matrix, dimensions_matrix + 2))

  #X-Values Matrix
  for i in range(dimensions_matrix):
    matrix[i][0] = x_points[i//2]
    matrix[i][1] = y_points[i//2]

  #Y-Values Matrix
  for i in range(1, dimensions_matrix, 2):
    matrix[i][2] = f_x[i//2]

  #F_x (derivatives) Matrix
  for i in range(2, dimensions_matrix):
    for j in range(2, i + 2):
      if matrix[i][j] != 0:
        continue
            
      matrix[i][j] = (matrix[i][j - 1] - matrix[i - 1][j - 1]) / (matrix[i][0] - matrix[i - j + 1][0])

  result = np.delete(matrix, -2, axis=1)
  print(result)

#Q5
def Cubic_Spline(x_points, y_points):
  dimensions_matrix = len(x_points)
  matrix = np.zeros((dimensions_matrix, dimensions_matrix))
  matrix[0][0] = matrix[dimensions_matrix - 1][dimensions_matrix - 1] = 1

  for i in range(1, dimensions_matrix - 1):
    index = i - 1
    for j in range(index, dimensions_matrix, 2):
      matrix[i][j] = x_points[index + 1] - x_points[index]
      index += 1

  for i in range(1, dimensions_matrix - 1):
    matrix[i][i] = 2 * ((x_points[i + 1] - x_points[i]) + (x_points[i] - x_points[i - 1]))

  print(np.matrix(matrix))
  print("")

  splines = np.zeros((dimensions_matrix))

  for i in range (1, dimensions_matrix - 1):
    first_term = (3/ (x_points[i + 1] - x_points[i])) * (y_points[i + 1] - y_points[i])
    second_term = (3/ (x_points[i] - x_points[i - 1])) * (y_points[i] - y_points[i - 1])
    splines[i] = first_term - second_term

  print(np.array(splines))
  print("")

#Use the built-in linear algebra function to solve for vector x 
  x= np.array(np.linalg.solve(matrix,splines))
#Print x Vector
  print(x)



if __name__ == "__main__":
#Input conditions Q1-Q5
  
# Q1 1 Neville's Method 
  x_points = [3.6, 3.8, 3.9]
  y_points = [1.675, 1.436, 1.318]
  f = 3.7
  print(Neville_Method(x_points, y_points, f))
  print("")

  # Question 2 Newton's Forward
  x_points = [7.2, 7.4, 7.5, 7.6]
  y_points = [23.5492, 25.3913, 26.8224, 27.4589]
  difference_table = Newtons_Forward(x_points, y_points)
  print("")

  #Q3 Approximate f(7.3)
  x0=7.3
  Approximation_Method(difference_table, x_points, x0, y_points[0])
  print("")

  #Question 4 Hermitian Approximation
  Hermite_Approximation([3.6, 3.8, 3.9], [1.675, 1.436, 1.318], [-1.195, -1.188, -1.182])
  print("")

  #Question 5 Cubic Spline Method
  Cubic_Spline([2, 5, 8, 10], [3, 5, 7, 9])
  print("")
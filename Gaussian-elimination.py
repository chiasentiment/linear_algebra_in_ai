# Databricks notebook source
import numpy as np

# COMMAND ----------

def swap_rows(M, row1, row2):
    '''Swap rows in a given matrix
    Parameters:
    M: array to swap(numpy array)
    row1: row to swap
    row2: row to swap
    Returns:
    M: array with swapped rows(numpy array)
    '''
    M=M.copy()
    M[[row1, row2]]= M[[row2, row1]]
    return M

# COMMAND ----------

def get_index_first_non_zero_value_column(M, column, start_row):
  '''
  retreive the index of the first non-zero column value
  input = M (matrix)
  column = column to check
  start_row = row to start the search
  return: index of the first non-zero value in that column
  '''
  for i, val in enumerate(M[start_row:,column]):
    if not np.close(val, 0, atol=1e-10):
      index = i+start_row
      return index
  return -1
      
 

# COMMAND ----------

def get_index_first_non_zero_value_row(M, row, augumented=False):
  '''
  retreive the index of the first non-zero row value
  input = M (matrix)
  row = row to check
  augumented = boolean to check if the matrix is augumented or not
  return: index of the first non-zero value in that row
  '''
  M = M.copy()
  if augumented:
    M = M[:, :-1]
  for i, val in enumerate(M[row]):
    if not np.isclose(val, 0, atol=1e-10):
      return i
  return -1

  

# COMMAND ----------

def augumented_matrix (A, B):
    '''
    create an augumented matrix from A and B
    by horizontally stacking two matrices
    input = A,B (numpy arrays)
    return: augumented matrix(numpy array)
    '''
    return np.hstack((A,B))

# COMMAND ----------

def row_echelon_form(A,B):
    '''
    convert a matrix to row echelon form
    input = M (numpy array)
    return: row echelon form(numpy array)
    '''
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0, atol=1e-10)== True:
        return "Singular Array"
    A = A.copy()
    B = B.copy()

    A = A.astype("float32")
    B = B.astype("float32")

    num_rows = len(A)
    M = augumented_matrix(A,B)
    for row in range(num_rows):
        pivot_candidate = M[row,row]

        if not np.isclose(pivot_candidate, 0) == True:
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_column(M, row, augumented=True)
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate)
            pivot = M[row, row]
        else:
            pivot = pivot_candidate
        M[row] = M[row]/pivot
        for j in range(row+1, num_rows):
            value_below_pivot = M[j, row]
            M[j] = M[j]-value_below_pivot*M[row]
    return M

# COMMAND ----------

A = np.array([[1,2,3],[0,1,0], [0,0,5]])
B = np.array([[1], [2], [4]])
row_echelon_form(A,B)


# COMMAND ----------

def back_substitution(M):
    '''
    perform back substitution on a matrix in row echelon form   
    '''
    M = M.copy()
    num_rows = M.copy()
    for row in reversed(range(num_rows)):
        substitution_row = M[row]
        index = get_index_first_non_zero_value_row(M, row, augumented=True)
        for j in range(row):
            row_to_reduce = M[j]
            value = row_to_reduce[index]
            row_to_reduce = row_to_reduce-value*substitution_row
            M[j,:] = row_to_reduce

    solution = M[:, -1]
    return solution

# COMMAND ----------

def gaussian_elimination(A,B):
    '''
    perform gaussian elimination on a matrix
    input = A,B (numpy arrays)
    return: solution(numpy array)
    '''
    M = row_echelon_form(A,B)
    solution = back_substitution(M)
    return solution

# COMMAND ----------

import numpy as np

# Coefficient matrix A
A = np.array([
    [2, 1],
    [1, 3]
])

# Constants matrix B
B = np.array([[8, 13]])

# Test the gaussian_elimination function
solution = gaussian_elimination(A, B)
display(solution)

# COMMAND ----------

equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""



# COMMAND ----------

x`

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



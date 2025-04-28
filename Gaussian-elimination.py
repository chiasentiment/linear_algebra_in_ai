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

:
  

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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



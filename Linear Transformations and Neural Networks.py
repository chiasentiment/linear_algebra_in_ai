# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC <a name="1"></a>
# MAGIC ## 1 - Introduction
# MAGIC
# MAGIC <a name="1.1"></a>
# MAGIC ### 1.1 - Transformations
# MAGIC
# MAGIC A **transformation** is a function from one vector space to another that respects the underlying (linear) structure of each vector space. Referring to a specific transformation, you can use a symbol, such as $T$. Specifying the spaces containing the input and output vectors, e.g. $\mathbb{R}^2$ and $\mathbb{R}^3$, you can write $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$. Transforming vector $v \in \mathbb{R}^2$ into the vector $w\in\mathbb{R}^3$ by the transformation $T$, you can use the notation $T(v)=w$ and read it as "*T of v equals to w*" or "*vector w is an **image** of vector v with the transformation T*".
# MAGIC
# MAGIC The following Python function corresponds to the transformation $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$ with the following symbolic formula:
# MAGIC
# MAGIC $$T\begin{pmatrix}
# MAGIC           \begin{bmatrix}
# MAGIC            v_1 \\           
# MAGIC            v_2
# MAGIC           \end{bmatrix}\end{pmatrix}=
# MAGIC           \begin{bmatrix}
# MAGIC            3v_1 \\
# MAGIC            0 \\
# MAGIC            -2v_2
# MAGIC           \end{bmatrix}
# MAGIC           \tag{1}
# MAGIC           $$

# COMMAND ----------

#Tranformations
def T(v):
    w = np.zeros((3,1))
    print (w)
    w[0,0] = 3*v[0,0]
    w[2,0]=-2*v[1,0]

    return w
v = np.array([[3],[5]])
w = T(v)
print ("Original vector: \n:", v, "\n Result of transformation\n", w)

# COMMAND ----------

#linear transformations
u = np.array([[1],[-2]])
v=np.array([[2],[4]])

k = 7

print ("T(k*v)\n: ",T(k*v))
print("k*T(v):",k*T(v))
print("T(u+v):\n", T(u+v), "\n\n T(u)+T(v):\n", T(u)+T(v))
            

# COMMAND ----------

#Transformation defined in Matrix multiplication
def L(v):
  A = np.array([[3,0],[0,0],[0,-2]])
  print ("Transformstion matrix: \n: ",A)
  w = A @ v
  return w
v = np.array([[3],[5]])
w = L(v)
print ("Original vector: \n:", v, "\n Result of transformation\n", w)


# COMMAND ----------

img = np.loadtxt("image.txt")
print ('shape: ',img.shape)
print (img)

# COMMAND ----------

plt.scatter(img[0], img[1], s = 0.001, color = 'black')

# COMMAND ----------

#horizondal scalling (Dilation)
def T_hscaling(v):
  A = np.array([[2,0],[0,1]])
  w = A @ v
  return w

def transform_vectors(T, v1, v2):
  V = np.hstack((v1, v2))
  W = T(V)

  return W
e1 = np.array([[1],[0]])
e2 = np.array([[0],[1]])
transformation_result_hscaling = transform_vectors(T_hscaling, e1, e2)
print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)

# COMMAND ----------

plt.scatter(img[0],img[1], s=0.001, color='black')
plt.scatter(T_hscaling(img)[0], T_hscaling(img)[1], s=0.001, color='red')

# COMMAND ----------

#Reflection about the y-axis

def T_reflection_yaxis(v):
    A = np.array([[-1,0],[0,1]])
    w = A @ v
    return w
e1 = np.array([[1],[0]])
e2 = np.array([[0],[1]])
transformation_result_reflection_yaxis = transform_vectors(T_reflection_yaxis, e1, e2)
print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_reflection_yaxis)

# COMMAND ----------

plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_reflection_yaxis(img)[0], T_reflection_yaxis(img)[1], s = 0.001, color = 'red')

# COMMAND ----------

#Streching by a scalar

def T_strech(a, v):
    T = np.array([[1,0],[0,1]])
    w = T@v*a
    return w
    

# COMMAND ----------

plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_strech(4,img)[0], T_strech(4,img)[1], s = 0.001, color = 'grey')

# COMMAND ----------



# COMMAND ----------






### backward_add_bias

The forward operation of tensor_add_bias operates with a $2 \times 2$ matrix as follows. We have a matrix and a bias vector,
$$
A = \begin{bmatrix}
        a_{11} & a_{12} \\
        a_{21} & a_{22} \\
        \end{bmatrix} 
        \quad b = \begin{bmatrix}
        b_1 & b_2\\
        \end{bmatrix}
$$

These are compatible because they have the same number of columns. We extend $b$ such that it is compatible with matrix addition, i.e.

$$
B = \begin{bmatrix}
        b_1 & b_2 \\
        b_1 & b_2 \\
    \end{bmatrix} \quad \Rightarrow \quad A + B = 
    \begin{bmatrix}
        a_{11} + b_1 & a_{12} + b_2 \\
        a_{21} + b_1 & a_{22} + b_2 \\
    \end{bmatrix}
$$

To understand how the gradient backpropogates through the operation, call $T = A + B$. We know the overall gradient of the loss $L$ with respect to $T$,

$$
\nabla_T = \begin{bmatrix}
        \frac{\partial L}{\partial t_{11}} & \frac{\partial L}{\partial t_{12}} \\
        \frac{\partial L}{\partial t_{21}} & \frac{\partial L}{\partial t_{22}} \\
        \end{bmatrix} \quad \text{ where } t_{ij} = a_{ij} + b_j 
$$

Then by the multivariable chain rule,

$$
\frac{\partial L}{\partial b_j} = \frac{\partial L}{\partial t_{1j}} \cdot \frac{\partial t_{1j}}{\partial b_j} + \frac{\partial L}{\partial t_{2j}} \cdot \frac{\partial t_{2j}}{\partial b_j}
$$

Note now, $\frac{\partial t_{ij}}{\partial b_j} =1$ always, so we end up just summing the overall gradients with respect to each $t_{ij}$, i.e.

$$
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial t_{11}} + \frac{\partial L}{\partial t_{21}} \quad \text{ and } \quad \frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial t_{12}} + \frac{\partial L}{\partial t_{22}}
$$

More generally, for an arbitrary size matrix such that the number of columns matches the length of the bias vector, letting k be the number of rows, i.e. the batch size, we get the general formula

$$
\frac{\partial L}{\partial b_j} = \sum_{i = 1}^k \frac{\partial L}{\partial t_{ij}}
$$

So 

$$
\nabla b = \left( \frac{\partial L}{\partial b_1}, \frac{\partial L}{\partial b_2}, \dots,
    \frac{\partial L}{\partial b_n} \right) = \left(\sum_{i = 1}^k \frac{\partial L}{\partial t_{i1}}, \sum_{i = 1}^k \frac{\partial L}{\partial t_{i2}}, \dots, \sum_{i = 1}^k \frac{\partial L}{\partial t_{in}}\right) = \sum_{i =1}^k \left(\frac{\partial L}{\partial t_{i1}}, \frac{\partial L}{\partial t_{i2}}, \dots, \frac{\partial L}{\partial t_{in}}\right)
$$

In code (for the cpu):

```c
if (bias->requires_grad) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < features; j++) {
            bias->cpu_grad[j] += t->cpu_grad[i * features + j];
        }
    }
}
```

It is easy to see how the gradient can be backpropogated through the matrix $A$ too. 

$$
\frac{\partial L}{\partial a_{ij}} = \frac{\partial L}{\partial t_{ij}} \cdot \frac{\partial t_{ij}}{\partial a_{ij}} = \frac{\partial L}{\partial t_{ij}} \quad \text{ since } \quad  \frac{\partial t_{ij}}{\partial a_{ij}} = 1
$$

Thus in code (for the cpu):

```c
if (a->requires_grad) {
    for (int i = 0; i < a->size; i++) {
        a->cpu_grad[i] += t->cpu_grad[i];
    }
}
```

### backward_matmul

The forward operation of tensor_matmul operates with $2 \times 2$ matrices as follows.

$$
A = 
\begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22} \\
\end{bmatrix} 
\quad B = 
\begin{bmatrix}
    b_{11} & b_{12} \\
    b_{21} & b_{22} \\
\end{bmatrix} \quad \Rightarrow AB = 
\begin{bmatrix}
    a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\
    a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22} \\
\end{bmatrix} 
$$

Now to understand how the gradient backpropogates through the operation, call $T = AB$. We know the overall gradient with respect to $T$,

$$
\nabla_T = \begin{bmatrix}
    \frac{\partial L}{\partial t_{11}} & \frac{\partial L}{\partial t_{12}} \\
    \frac{\partial L}{\partial t_{21}} & \frac{\partial L}{\partial t_{22}} \\
    \end{bmatrix} \quad \text{ where } t_{ij} = \sum_{k = 1}^n a_{ik}b_{kj}
$$

Then by the multivariable chain rule,

$$
\frac{\partial L}{\partial a_{11}} = \frac{\partial L}{\partial t_{11}} \cdot \frac{\partial t_{11}}{\partial a_{11}} + \frac{\partial L}{\partial t_{12}} \cdot \frac{\partial t_{12}}{\partial a_{11}} \quad \text{ and } \quad
\frac{\partial L}{\partial a_{12}} = \frac{\partial L}{\partial t_{11}} \cdot \frac{\partial t_{11}}{\partial a_{12}} + \frac{\partial L}{\partial t_{12}} \cdot \frac{\partial t_{12}}{\partial a_{12}}
$$
and
$$
\frac{\partial L}{\partial a_{21}} = \frac{\partial L}{\partial t_{21}} \cdot \frac{\partial t_{21}}{\partial a_{21}} + \frac{\partial L}{\partial t_{21}} \cdot \frac{\partial t_{21}}{\partial a_{21}} \quad \text{ and } \quad
\frac{\partial L}{\partial a_{22}} = \frac{\partial L}{\partial t_{21}} \cdot \frac{\partial t_{21}}{\partial a_{22}} + \frac{\partial L}{\partial t_{22}} \cdot \frac{\partial t_{22}}{\partial a_{22}}
$$

Noticing the pattern,

$$
\frac{\partial L}{\partial a_{ij}} = \sum_{n = 1}^N \left(\frac{\partial L}{\partial t_{in}}\right) \left(\frac{\partial t_{in}}{\partial a_{ij}}\right) = \sum_{n = 1}^N \nabla T_{in} \cdot b_{jn}
$$

Now notice that $b_{jn} = (B^T)_{nj}$. Thus, substituting this in

$$
\frac{\partial L}{\partial a_{ij}} = \sum_{n = 1}^N \nabla T_{in} \cdot b_{jn} = 
\sum_{n = 1}^N \nabla T_{in} \cdot (B^T)_{nj} \quad \Rightarrow \quad \nabla_A= \nabla_T  B^T
$$

With similar reasoning $\nabla_B = A^T \nabla_T$.

Thus in code (for the cpu):


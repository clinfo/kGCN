import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops


class BatchedSpMM:
    def __init__(self):
        self.b_module = tf.load_op_library('./batched.so')

    def call(self, sp_matrices, dense_matrices, adjoint_a=False, adjoint_b=False):
        sp_indices = [sp_m.indices for sp_m in sp_matrices]
        sp_values = [sp_m.values for sp_m in sp_matrices]
        sp_shape = [sp_m.dense_shape for sp_m in sp_matrices]
        return self.b_module.bspmm(sp_ids = sp_indices, sp_values = sp_values, sp_shape = sp_shape, rhs = dense_matrices, adjoint_a = adjoint_a, adjoint_b = adjoint_b)


class BatchedSpMDT:
    def __init__(self):
        self.b_module = tf.load_op_library('./batched.so')

    def call(self, sp_matrices, dense_matrices, adjoint_a=False, adjoint_b=False):
        sp_indices = [sp_m.indices for sp_m in sp_matrices]
        sp_values = [sp_m.values for sp_m in sp_matrices]
        sp_shape = [sp_m.dense_shape for sp_m in sp_matrices]

        return self.b_module.bspmdt(sp_ids = sp_indices, sp_values = sp_values, sp_shape = sp_shape, rhs = dense_matrices, adjoint_a = adjoint_a, adjoint_b = adjoint_b)


b_module = tf.load_op_library('./batched.so')


@ops.RegisterGradient("Bspmdt")
def _bspmdt_grad(op, *grad):
    """Gradients for the dense tensors in the SparseTensorDenseMatMul ops.
    Args:
    op: the Bspmdt ops
    grads: the incoming gradients

    Returns:
    Gradients for each of the 4 input tensors:
    (sparse_indices, sparse_values, sparse_shape, dense_tensor)
    The gradients for indices and shape are None.

    """
    numTensors = (len(op.inputs) - 1) // 3
  
    a_indices = op.inputs[0:numTensors]
    a_values = op.inputs[numTensors:numTensors*2]
    a_shape = op.inputs[numTensors*2:numTensors*3]
    b = op.inputs[numTensors*3]
    adj_a = op.get_attr("adjoint_a")
    adj_b = op.get_attr("adjoint_b")
  
    # gradient w.r.t. dense
    a_values_grads = []
    b_list = [b[i] for i in range(numTensors)]
  
    b_grads = b_module.bspmm(a_indices, a_values, a_shape, grad, adjoint_a=True, adjoint_b=False)

    bg_row=tf.shape(input=b_grads[0])[0]
    bg_col=tf.shape(input=b_grads[0])[1]
    b_grads = tf.reshape(b_grads, (numTensors * bg_row, bg_col))

    if adj_b:
        b_grads = [array_ops.transpose(b_g) for b_g in b_grads]
  
    for t in range(numTensors):
        rows = a_indices[t][:, 0]
        cols = a_indices[t][:, 1]
        parts_a = array_ops.gather(grad[t], rows if not adj_a else cols)
        parts_b = array_ops.gather(b_list[t] if not adj_b else array_ops.transpose(b_list[t]), cols if not adj_a else rows)
        a_values_grads.append(math_ops.reduce_sum(parts_a * parts_b, reduction_indices=1))

    return_val = [None for _ in range(numTensors)] + a_values_grads + [None for _ in range(numTensors)] + [b_grads]
    return tuple(return_val)

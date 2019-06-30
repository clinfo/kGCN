import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

class BatchedConv:
	def __init__(self):
		self.bconv_module = tf.load_op_library('./bconv.so')

	def call(self, sp_matrices, dense_matrices, adjoint_a=False, adjoint_b=False):
		sp_m = [sp_m for sp_ms in sp_matrices for sp_m in sp_ms]
		sp_indices = [m.indices for m in sp_m]
		sp_values = [m.values for m in sp_m]
		sp_shape = [m.dense_shape for m in sp_m]
		rhs = [dm for dms in dense_matrices for dm in dms]
		
		batchSize = len(dense_matrices)
		numChannels = len(dense_matrices[0])
		dim_matrices = [numChannels, batchSize]
		
		return self.bconv_module.bconv(dim_matrices=dim_matrices, sp_ids = sp_indices, sp_values = sp_values, sp_shape = sp_shape, rhs = rhs, adjoint_a = adjoint_a, adjoint_b = adjoint_b)

bconv_module = tf.load_op_library('./bconv.so')
bspmm_module = tf.load_op_library('./bspmm.so')

@ops.RegisterGradient("Bconv")
def _bconv_grad(op, *grad):

  """Gradients for the dense tensors in the SparseTensorDenseMatMul ops.
  Args:
    op: the Bconv ops
    grads: the incoming gradients

  Returns:
  Gradients for each of the 5 input tensors:
      (dim_matrices, sparse_indices, sparse_values, sparse_shapes, dense_tensors)
    The gradients for indices and shape are None.
  """
  numTensors = (len(op.inputs) - 1) // 4
  batchSize = len(op.outputs)
  numChannels = numTensors // batchSize

  # addn_grad = [grad] * numChannels
  addn_grad = [g for g in grad for _ in range(numChannels)]

  dim_matrices = op.inputs[0]
  a_indices = op.inputs[1:numTensors + 1]
  a_values = op.inputs[numTensors+1:numTensors*2+1]
  a_shape = op.inputs[numTensors*2+1:numTensors*3+1]
  b = op.inputs[numTensors*3+1:numTensors*4+1]
  adj_a = op.get_attr("adjoint_a")
  adj_b = op.get_attr("adjoint_b")
  
  a_values_grads = []
  b_grads = bspmm_module.bspmm(a_indices, a_values, a_shape, addn_grad, adjoint_a=not adj_a, adjoint_b=adj_b)

  if adj_b:
    b_grads = [array_ops.transpose(b_g) for b_g in b_grads]
  
  for t in range(numTensors):
    rows = a_indices[t][:, 0]
    cols = a_indices[t][:, 1]
    parts_a = array_ops.gather(addn_grad[t], rows if not adj_a else cols)
    parts_b = array_ops.gather(b[t] if not adj_b else array_ops.transpose(b[t]), cols if not adj_a else rows)
    a_values_grads.append(math_ops.reduce_sum(parts_a * parts_b, reduction_indices=1))

	
  return_val = [None] + [None for _ in range(numTensors)] + a_values_grads + [None for _ in range(numTensors)] + b_grads
  return tuple(return_val)

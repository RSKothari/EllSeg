/.autofs/tools/spack/opt/spack/linux-rhel7-x86_64/gcc-7.4.0/py-torch-1.2.0-fdksqtbpsau7tm3ql4lsmqej7rf5tgov/lib/python3.6/site-packages/torch/nn/parallel/_functions.py:61: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
Traceback (most recent call last):
  File "train.py", line 147, in <module>
    loss.backward()
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-x86_64/gcc-7.4.0/py-torch-1.2.0-fdksqtbpsau7tm3ql4lsmqej7rf5tgov/lib/python3.6/site-packages/torch/tensor.py", line 118, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-x86_64/gcc-7.4.0/py-torch-1.2.0-fdksqtbpsau7tm3ql4lsmqej7rf5tgov/lib/python3.6/site-packages/torch/autograd/__init__.py", line 87, in backward
    grad_tensors = _make_grads(tensors, grad_tensors)
  File "/.autofs/tools/spack/opt/spack/linux-rhel7-x86_64/gcc-7.4.0/py-torch-1.2.0-fdksqtbpsau7tm3ql4lsmqej7rf5tgov/lib/python3.6/site-packages/torch/autograd/__init__.py", line 28, in _make_grads
    raise RuntimeError("grad can be implicitly created only for scalar outputs")
RuntimeError: grad can be implicitly created only for scalar outputs

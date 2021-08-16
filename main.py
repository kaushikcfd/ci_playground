import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

SRC = """
__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) _pt_kernel_0(
    __global long const *__restrict__ _actx_dw,
    __global long const *__restrict__ _actx_dw_0,
    __global long *__restrict__ _pt_out,
    __global long *__restrict__ _pt_temp)
{
  for (int comp_iel_5 = 0; comp_iel_5 <= 3; ++comp_iel_5)
    _pt_temp[_actx_dw_0[comp_iel_5]] = _actx_dw[comp_iel_5];
  for (int _pt_out_dim0 = 0; _pt_out_dim0 <= 3; ++_pt_out_dim0)
    _pt_out[_pt_out_dim0] = _pt_temp[_pt_out_dim0];
}
"""


dw = np.arange(4, dtype=np.int64)*8
dw_0 = np.arange(4, dtype=np.int64)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dw_dev = cl_array.to_device(queue, dw)
dw_0_dev = cl_array.to_device(queue, dw_0)
temp_dev = cl_array.to_device(queue, dw*0 - 1)
out_dev = cl_array.empty_like(dw_dev)

prg = cl.Program(ctx, SRC).build(options=["-cl-opt-disable"])

prg._pt_kernel_0(
        queue, (1,), (1,), dw_dev.data, dw_0_dev.data, out_dev.data,
        temp_dev.data)
print("Output =", out_dev)
np.testing.assert_allclose(out_dev.get(),
                           [0, 8, 16, 24])
print("Yay passed!!!")

################################################################################
# Example 2: Manually Optimizing Matrix Multiplication with TE
# ------------------------------------------------------------
#
# Now we will consider a second, more advanced example, demonstrating how with
# just 18 lines of python code TVM speeds up a common matrix multiplication operation by 18x.
#
# **Matrix multiplication is a compute intensive operation. There are
# two important optimizations for good CPU performance:**
#
# 1. Increase the cache hit rate of memory access. Both complex
#    numerical computation and hot-spot memory access can be
#    accelerated by a high cache hit rate. This requires us to
#    transform the origin memory access pattern to a pattern that fits
#    the cache policy.
#
# 2. SIMD (Single instruction multi-data), also known as the vector
#    processing unit. On each cycle instead of processing a single
#    value, SIMD can process a small batch of data.  This requires us
#    to transform the data access pattern in the loop body in uniform
#    pattern so that the LLVM backend can lower it to SIMD.
#
# The techniques used in this tutorial are a subset of tricks mentioned in this
# `repository <https://github.com/flame/how-to-optimize-gemm>`_. Some of them
# have been applied by TVM abstraction automatically, but some of them cannot
# be automatically applied due to TVM constraints.

################################################################################
# Preparation and Performance Baseline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We begin by collecting performance data on the `numpy` implementation of
# matrix multiplication.

import tvm
import tvm.testing
from tvm import te
import numpy

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor data type in tvm
dtype = "float32"

# You will want to adjust the target to match any CPU vector extensions you
# might have. For example, if you're using using Intel AVX2 (Advanced Vector
# Extensions) ISA for SIMD, you can get the best performance by changing the
# following line to ``llvm -mcpu=core-avx2``, or specific type of CPU you use.
# Recall that you're using llvm, you can get this information from the command
# ``llc --version`` to get the CPU type, and you can check ``/proc/cpuinfo``
# for additional extensions that your processor might support.

target = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(target.kind.name, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

# Repeatedly perform a matrix multiplication to get a performance baseline
# for the default numpy implementation
np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

################################################################################
# Now we write a basic matrix multiplication using TVM TE and verify that it
# produces the same results as the numpy implementation. We also write a
# function that will help us measure the performance of the schedule
# optimizations.

# TVM Matrix Multiplication using TE
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name="mmult")

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)


def evaluate_operation(s, vars, target, name, optimization, log):
    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))
    log.append((optimization, mean_time))


log = []

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="none", log=log)

################################################################################
# Let's take a look at the intermediate representation of the operator and
# default schedule using the TVM lower function. Note how the implementation is
# essentially a naive implementation of a matrix multiplication, using three
# nested loops over the indices of the A and B matrices.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 1: Blocking
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# A important trick to enhance the cache hit rate is blocking, where you
# structure memory access such that the inside a block is a small neighborhood
# that has high memory locality. In this tutorial, we pick a block factor of
# 32. This will result in a block that will fill a 32 * 32 * sizeof(float) area
# of memory. This corresponds to a cache size of 4KB, in relation to a
# reference cache size of 32 KB for L1 cache.
#
# We begin by creating a default schedule for the ``C`` operation, then apply a
# ``tile`` scheduling primitive to it with the specified block factor, with the
# scheduling primitive returning the resulting loop order from outermost to
# innermost, as a vector ``[x_outer, y_outer, x_inner, y_inner]``. We then get
# the reduction axis for output of the operation, and perform a split operation
# on it using a factor of 4. This factor doesn't directly impact the blocking
# optimization we're working on right now, but will be useful later when we
# apply vectorization.
#
# Now that the operation has been blocked, we can reorder the computation to
# put the reduction operation into the outermost loop of the computation,
# helping to guarantee that the blocked data remains in cache. This completes
# the schedule, and we can build and test the performance compared to the naive
# schedule.

bn = 32

# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# Hoist reduction domain outside the blocking loop
s[C].reorder(xo, yo, ko, ki, xi, yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="blocking", log=log)

################################################################################
# By reordering the computation to take advantage of caching, you should see a
# significant improvement in the performance of the computation. Now, print the
# internal representation and compare it to the original:

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 2: Vectorization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Another important optimization trick is vectorization. When the memory access
# pattern is uniform, the compiler can detect this pattern and pass the
# continuous memory to the SIMD vector processor. In TVM, we can use the
# ``vectorize`` interface to hint the compiler this pattern, taking advantage
# of this hardware feature.
#
# In this tutorial, we chose to vectorize the inner loop row data since it is
# already cache friendly from our previous optimizations.

# Apply the vectorization optimization
s[C].vectorize(yi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="vectorization", log=log)

# The generalized IR after vectorization
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 3: Loop Permutation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If we look at the above IR, we can see the inner loop row data is vectorized
# and B is transformed into PackedB (this is evident by the `(float32x32*)B2`
# portion of the inner loop). The traversal of PackedB is sequential now. So we
# will look at the access pattern of A. In current schedule, A is accessed
# column by column which is not cache friendly. If we change the nested loop
# order of `ki` and inner axes `xi`, the access pattern for A matrix will be
# more cache friendly.

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# re-ordering
s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="loop permutation", log=log
)

# Again, print the new generalized IR
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 4: Array Packing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Another important trick is array packing. This trick is to reorder the
# storage dimension of the array to convert the continuous access pattern on
# certain dimension to a sequential pattern after flattening.
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/array-packing.png
#    :align: center
#
# Just as it is shown in the figure above, after blocking the computations, we
# can observe the array access pattern of B (after flattening), which is
# regular but discontinuous. We expect that after some transformation we can
# get a continuous access pattern. By reordering a ``[16][16]`` array to a
# ``[16/4][16][4]`` array the access pattern of B will be sequential when
# grabing the corresponding value from the packed array.
#
# To accomplish this, we are going to have to start with a new default
# schedule, taking into account the new packing of B. It's worth taking a
# moment to comment on this: TE is a powerful and expressive language for
# writing optimized operators, but it often requires some knowledge of the
# underlying algorithm, data structures, and hardware target that you are
# writing for. Later in the tutorial, we will discuss some of the options for
# letting TVM take that burden. Regardless, let's move on with the new
# optimized schedule.

# We have to re-write the algorithm slightly.
packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="array packing", log=log)

# Here is the generated IR after array packing.
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 5: Optimizing Block Writing Through Caching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Up to this point all of our optimizations have focused on efficiently
# accessing and computing the data from the `A` and `B` matrices to compute the
# `C` matrix. After the blocking optimization, the operator will write result
# to `C` block by block, and the access pattern is not sequential. We can
# address this by using a sequential cache array, using a combination of
# `cache_write`, `compute_at`, and `unroll`to hold the block results and write
# to `C` when all the block results are ready.

s = te.create_schedule(C.op)

# Allocate write cache
CC = s.cache_write(C, "global")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# Write cache is computed at yo
s[CC].compute_at(s[C], yo)

# New inner axes
xc, yc = s[CC].op.axis

(k,) = s[CC].op.reduce_axis
ko, ki = s[CC].split(k, factor=4)
s[CC].reorder(ko, xc, ki, yc)
s[CC].unroll(ki)
s[CC].vectorize(yc)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="block caching", log=log)

# Here is the generated IR after write cache blocking.
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Optimization 6: Parallelization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# So far, our computation is only designed to use a single core. Nearly all
# modern processors have multiple cores, and computation can benefit from
# running computations in parallel. The final optimization is to take advantage
# of thread-level parallelization.

# parallel
s[C].parallel(xo)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(
    s, [A, B, C], target=target, name="mmult", optimization="parallelization", log=log
)

# Here is the generated IR after parallelization.
print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################
# Summary of Matrix Multiplication Example
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# After applying the above simple optimizations with only 18 lines of code, our
# generated code can begin to approach the performance of `numpy` with the Math
# Kernel Library (MKL). Since we've been logging the performance as we've been
# working, we can compare the results.

baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )

################################################################################
# Note that the outputs on the web page reflect the running times on a
# non-exclusive Docker container, and should be considered unreliable. It is
# highly encouraged to run the tutorial by yourself to observe the performance
# gain achieved by TVM, and to carefully work through each example to
# understand the iterative improvements that are made to the matrix
# multiplication operation.

################################################################################
# Final Notes and Summary
# -----------------------
# As mentioned earlier, how to apply optimizations using TE and scheduling
# primitives can require some knowledge of the underlying architecture and
# algorithms. However, TE was designed to act as a foundation for more complex
# algorithms that can search the potential optimization. With the knowledge you
# have from this introduction to TE, we can now begin to explore how TVM can
# automate the schedule optimization process.
#
# This tutorial provided a walkthrough of TVM Tensor Expresstion (TE) workflow
# using a vector add and a matrix multiplication examples. The general workflow
# is
#
# - Describe your computation via a series of operations.
# - Describe how we want to compute use schedule primitives.
# - Compile to the target function we want.
# - Optionally, save the function to be loaded later.
#
# Upcoming tutorials expand on the matrix multiplication example, and show how
# you can build generic templates of the matrix multiplication and other
# operations with tunable parameters that allows you to automatically optimize
# the computation for specific platforms.


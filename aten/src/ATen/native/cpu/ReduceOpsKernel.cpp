#include <numeric>
#include <iterator>
#include <algorithm>
#include <limits>
#include <iostream>

#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <c10/util/Optional.h>

namespace at { namespace native { namespace {

using namespace vec256;

static void sum_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Bool, iter.dtype(), "sum_cpu", [&] {
        binary_kernel_reduce_vec(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
            [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a + b; });
      });
}

static void mean_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "mean_cpu", [&] {
    scalar_t factor = scalar_t(iter.num_output_elements()) / iter.numel();
    binary_kernel_reduce(
      iter,
      MeanOps<scalar_t, scalar_t> {factor},
      scalar_t(0)
    );
  });
}

static void std_var_kernel_impl(TensorIterator &iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std_cpu", [&] {
      std::cout << "calling std var\n";
    binary_kernel_reduce(
      iter,
      WelfordOps<scalar_t, double, int64_t, double, std::tuple<scalar_t, scalar_t>> { unbiased, take_sqrt },
      WelfordData<double, int64_t, double>()
    );
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "prod_cpu", [&] {
    binary_kernel_reduce_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a * b; },
      /*identity=*/1);
  });
}

static void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    Scalar p) {
  float val;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<float>();
  } else {
    AT_ERROR("norm_kernel_tensor_iterator_impl expects norm to be integer or float");
  }


  if (val == 0) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        NormZeroOps<scalar_t>(),
        scalar_t(0)
      );
    });
  } else if (val == 1) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        NormOneOps<scalar_t>(),
        scalar_t(0)
      );
    });
  } else if (val == INFINITY) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        AbsMaxOps<scalar_t>(),
        std::numeric_limits<scalar_t>::min()
      );
    });
  } else if (val == -INFINITY) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        AbsMinOps<scalar_t>(),
        std::numeric_limits<scalar_t>::max()
      );
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        NormOps<scalar_t> { scalar_t(val) },
        scalar_t(0)
      );
    });
  }
}

static void and_kernel_impl(TensorIterator& iter) {
  binary_kernel_reduce_vec(
    iter,
    [=](uint8_t a, uint8_t b) -> uint8_t { return a && b; },
    [=](Vec256<uint8_t> a, Vec256<uint8_t> b) {
      // Adding the implementation here instead of in vec256_base to avoid
      // return value inconsistency. Other comparison operators in vec256_base
      // return -1/0 (all bit 1 / all bit 0) as true/false to follow the AVX2
      // convention. This would be convenient when combined with other
      // vectorized operations. For example, one can use the logical operation
      // results as a mask for a bit operation to retrieve/reset multiple
      // elements in a vector.
      //
      // In this method, users would expect, e.g., all(), to return 1/0 as
      // true/false.
      Vec256<uint8_t> c = Vec256<uint8_t>();
      for (int i = 0; i != Vec256<uint8_t>::size(); i++) {
        c[i] = a[i] && b[i];
      }
      return c;
    },
    /*ident=*/true);
}

static void or_kernel_impl(TensorIterator& iter) {
  binary_kernel_reduce_vec(
    iter,
    [=](uint8_t a, uint8_t b) -> uint8_t { return a || b; },
    [=](Vec256<uint8_t> a, Vec256<uint8_t> b) {
      Vec256<uint8_t> c = Vec256<uint8_t>();
      for (int i = 0; i != Vec256<uint8_t>::size(); i++) {
        c[i] = a[i] || b[i];
      }
      return c;
    },
    /*ident=*/false);
}

static void min_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "min_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); },
      [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return minimum(a, b); });
  });
}

static void max_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "max_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return std::max(a, b); },
      [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return maximum(a, b); });
  });
}

// Maximum and minimum possible scalar values, including infinities

template <typename scalar_t>
constexpr scalar_t upper_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? lim::infinity() : lim::max();
}

template <typename scalar_t>
constexpr scalar_t lower_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? -lim::infinity() : lim::lowest();
}

static void argmax_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_cpu", [&] {
    binary_kernel_reduce(
      iter,
      ArgMaxOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(lower_bound<scalar_t>(), -1));
  });
}

static void argmin_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_cpu", [&] {
    binary_kernel_reduce(
      iter,
      ArgMinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
  });
}

static void cumsum_kernel_impl(TensorIterator &iter) {
  int dim = 0; 
    AT_DISPATCH_ALL_TYPES(
      iter.dtype(), "cumsum", [&] {
        scalar_t cumsum = 0;
        int reduce_dims = 1;
        // get number of dimensions to reduce. we do it this way in this
        // case since we don't have a 0 stride dimension which can be taken
        // as the 'reduction' dimension. Therefore we multiply the dimensions
        // except the dimension over which the user wants the cumsum in order
        // to find the number of iterations to perform.
        for (int i = 0; i < iter.ndim(); ++i) {
          if (i != dim) {
            reduce_dims *= iter.shape()[i];
          }
        }

        std::cout << "output shape :: " << iter.output().sizes() << std::endl;
        std::cout << "reduce_dims: " << reduce_dims << std::endl;
        int non_reduce_dim = dim +1;

        auto non_reduced_shape = iter.output().sizes().slice(reduce_dims,
                                                             iter.output().sizes().size() - reduce_dims);
        int64_t non_reduced_numel = 1;
        for (int i = 0; i < non_reduced_shape.size(); ++i) {
          non_reduced_numel *= non_reduced_shape[i];
        }

        std::cout << "non reduced numel: " << non_reduced_numel << std::endl;
        std::cout << "non reduced shape: " << non_reduced_shape << std::endl;

        DimCounter dims {non_reduced_shape, {0, non_reduced_numel}};

        while (!dims.is_done()) {
          std::cout << "while dims: " << dims.values << std::endl;
          dims.increment({1,1});
        }

        std::cout << "------- new stuff ----\n";
        std::vector<int64_t> d = iter.output().sizes().vec();
        d[dim] = 0;
        non_reduced_shape = IntArrayRef(d);
        // std::cout << ""
        // non_reduced_shape = IntArrayRef(iter.output().sizes());
        std::cout << "nonreduced shape: " << non_reduced_shape << std::endl;
        // non_reduced_shape[dim] = 0;

        non_reduced_numel = 1;
        for (int i = 0 ; i < non_reduced_shape.size(); i++) {
          if (i != dim)
            non_reduced_numel *= non_reduced_shape[i];
        }

        DimCounter newdims {non_reduced_shape, {0, non_reduced_numel}};
        std::cout << "non reduced numel: " << non_reduced_numel << std::endl;

        std::cout << "newdims: " << newdims.values << std::endl;
        while(!newdims.is_done()) {
          std::cout << "newdims: " << newdims.values << std::endl;
          newdims.increment({1,1});
        }
        
        if (iter.numel() < internal::GRAIN_SIZE) {
          at::native::cpu_serial_kernel(iter,
                                        [&](scalar_t input) {
                                          cumsum += input;
                                          return cumsum;
                                        });
        }
        else {
          // parallel implementation.
          std::cout << "no parallel impl yet.";
        }
      });
}

}  // anonymous namespace

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);
REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);
REGISTER_DISPATCH(mean_stub, &mean_kernel_impl);
REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl);
REGISTER_DISPATCH(and_stub, &and_kernel_impl);
REGISTER_DISPATCH(or_stub, &or_kernel_impl);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_impl);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl);
REGISTER_DISPATCH(cumsum_stub, &cumsum_kernel_impl);

}}  // namespace at::native

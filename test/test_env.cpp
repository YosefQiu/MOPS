#include <sycl/sycl.hpp>

#include <cmath>
#include <iostream>
#include "ndarray/ndarray_group_stream.hh"
int main() {
  const int n = 100000;

  sycl::buffer<double> b_a{n}, b_b{n}, b_c{n};

  {
    sycl::host_accessor a{b_a, sycl::write_only};
    sycl::host_accessor b{b_b, sycl::write_only};
    for (size_t i = 0; i < n; i++) {
      a[i] = sin(i) * sin(i);
      b[i] = cos(i) * cos(i);
    }
  }

  sycl::queue q{sycl::gpu_selector_v};

  q.submit([&](sycl::handler &h) {
    sycl::accessor a{b_a, h, sycl::read_only};
    sycl::accessor b{b_b, h, sycl::read_only};
    sycl::accessor c{b_c, h, sycl::write_only};

    h.parallel_for(n, [=](sycl::id<1> i) { c[i] = a[i] + b[i]; });
  });

  {
    double sum = 0.0;
    sycl::host_accessor c{b_c, sycl::read_only};
    for (size_t i = 0; i < n; i++)
      sum += c[i];
    std::cout << "sum = " << sum / n << std::endl;

    if (!(fabs(sum - static_cast<double>(n)) <= 1.0e-8))
      return 1;
  }

  return 0;
}

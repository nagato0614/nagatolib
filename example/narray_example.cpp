//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"
using namespace nagato;

template<typename ArrayType>
auto step_function(const ArrayType &x)
{
  return na::AsType<int>(x > 0);
}

int main(int argc, char *argv[])
{
  using namespace nagato;

  // concatenate
  auto a = na::Fill<float, 2>(1.f);
  auto b = na::Fill<float, 2>(2.f);

  // NagatoArray<float, 4> の配列になる
  auto c = na::Concatenate1D(a, b);
  na::Show(c);

  auto d = na::Fill<float, 2, 2>(1);
  auto e = na::Fill<float, 2, 2>(2);

  // NagatoArray<float, 4, 2> の配列になる
  auto f = na::ConcatenateND<4, 2>(d, e);

  na::Show(f);

  return 0;
}
//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"

int main(int argc, char *argv[])
{
  using namespace nagato;

  const auto a = na::NagatoArray<int, 3, 3, 3>(1);
  const auto b = na::NagatoArray<int, 3, 3, 3>(2);

  const auto c = na::Transform(a, b, [](int x, int y) { return x + y; });

  return 0;
}
//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"

int main(int argc, char *argv[])
{
  using namespace nagato;

  const auto a = na::Array<float, 2, 2>(1.f);

  std::cout << a(0, 0) << std::endl;

  const na::NagatoArrayInner<float, 2> b = a[0];

  std::cout << b(0) << std::endl;

  std::cout << a[0][0] << std::endl;

  a[0][0] = 2.f;

  std::cout << a[0][0] << std::endl;

  return 0;
}
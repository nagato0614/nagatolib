//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"

int main(int argc, char *argv[])
{
  using namespace nagato;

  auto a = na::Fill<float, 4, 4>(1.f);

  std::cout << a(0, 0) << std::endl;

  auto b = a[0];

  std::cout << b(0) << std::endl;

  std::cout << a[0][0] << std::endl;

  a[0][0] = 2.f;

  std::cout << a[0][0] << std::endl;

  na::Show(a);

  auto c = na::Fill<float, 4>(1.f);

  std::cout << c[0] << std::endl;
  c[1] = 2.f;
  na::Show(c);

  const na::NagatoArray<float, 4> d = {1.f, 2.f, 3.f, 4.f};

  na::Show(d);

  return 0;
}
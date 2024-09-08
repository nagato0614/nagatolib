//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"

int main(int argc, char *argv[])
{
  using namespace nagato;

  auto a = na::Fill<float, 4, 4>(1.5f);
  auto b = na::Fill<float, 4, 4>(2.5f);

  b[0][0] = 1.5f;

  auto c = a < b;
  na::Show(c);

  return 0;
}
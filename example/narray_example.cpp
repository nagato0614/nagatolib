//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"

int main(int argc, char *argv[])
{
  using namespace nagato;

  const auto a = na::NagatoArray<int, 1, 1, 1>(1);

  na::Show(a);
  return 0;
}
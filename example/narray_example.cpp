//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"

int main(int argc, char *argv[])
{
  using namespace nagato;

  const auto a = na::NagatoArray<float, 3, 3, 3>(4);

  const auto c = na::Transform(a,
                               [](auto v)
                               { return v * 2; }
  );
  na::Show(c);

  return 0;
}
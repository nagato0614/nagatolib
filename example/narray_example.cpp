//
// Created by toru on 2024/09/02.
//
#include <iostream>

#include "narray.hpp"

int main(int argc, char *argv[])
{
  using namespace nagato;

  const auto a = na::NagatoArray<float, 3, 3, 3>(4);

  const auto c = na::Transform(a, 10,
                               [](auto v, auto w)
                               { return v + w; }
  );
  na::Show(c);

  return 0;
}
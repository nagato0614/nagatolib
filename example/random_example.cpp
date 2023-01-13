//
// Created by nagato0614 on 2019-07-12.
//
#include <iostream>
#include <random.hpp>

constexpr double avg(int N) {
  nagato::LinearCongruential<unsigned int> test;

  double sum = 0;
  for (int i = 0; i < N; i++) {
    sum += test.uniform_int<int>(0, 100);
  }

  return sum / N;
}

int main() {

  constexpr int N = 100000000;

  std::cout << avg(N) << std::endl;
}

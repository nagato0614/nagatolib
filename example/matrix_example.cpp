//
// Created by nagato0614 on 2019-07-14.
//
#include <iostream>

#include "matrix.hpp"

int main() {
  using mat3d = nagato::Matrix<double, 3, 3>;
  using vec3d = nagato::Vector<double, 3>;

  constexpr mat3d m(0);
  m.Max();
  m.Min();

  std::cout << m << std::endl;

  constexpr mat3d m_1 = {{1, 2, 3},
						 {4, 5, 6},
						 {7, 8, 9}};
  constexpr vec3d v_1 = {1, 2, 3};

  std::cout << m_1 << std::endl;
  std::cout << m_1 + m_1 << std::endl;
  std::cout << m_1 * mat3d::Identity() << std::endl;
  std::cout << m_1 * v_1 << std::endl;
  std::cout << v_1 * m_1 << std::endl;
}
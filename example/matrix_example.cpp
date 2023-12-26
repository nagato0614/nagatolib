//
// Created by nagato0614 on 2019-07-14.
//
#include <iostream>

#include "matrix.hpp"

int main()
{
  using mat3d = nagato::Matrix<double, 3, 3>;
  using mat2d = nagato::Matrix<double, 2, 2>;
  using vec3d = nagato::Vector<double, 3>;
  using mat32d = nagato::Matrix<double, 3, 2>;
  using mat23d = nagato::Matrix<double, 2, 3>;

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

  constexpr mat2d x = {
    {1, 2},
    {3, 4},
  };

  constexpr mat2d y = {
    {5, 6},
    {7, 8},
  };

  std::cout << "x * y" << std::endl;
  std::cout << x * y << std::endl;

  const mat32d m32 = {
    {1, 2},
    {3, 4},
    {5, 6},
  };

  const mat23d m23 = {
    {1, 2, 3},
    {4, 5, 6},
  };

  std::cout << "m32, m23" << std::endl;
  std::cout << m32 << std::endl;
  std::cout << m23 << std::endl;

  std::cout << "---" << std::endl;

  const mat3d m32m23 = m32 * m23;
  std::cout << "m32 * m23" << std::endl;
  std::cout << m32m23 << std::endl;

  const mat2d m23m32 = m23 * m32;
  std::cout << "m23 * m32" << std::endl;
  std::cout << m23m32 << std::endl;

}
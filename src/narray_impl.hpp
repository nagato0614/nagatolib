//
// Created by toru on 2024/09/02.
//

#ifndef NAGATOLIB_METAL_NARRAY_IMPL_HPP_
#define NAGATOLIB_METAL_NARRAY_IMPL_HPP_

#include "narray.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace nagato::mla
{
template<typename T>
constexpr bool is_metal_arithmetic_type()
{
  bool type_list[] =
    {
      std::is_same<T, bool>::value,
      std::is_same<T, char>::value,
      std::is_same<T, unsigned char>::value,
      std::is_same<T, short>::value,
      std::is_same<T, unsigned short>::value,
      std::is_same<T, int>::value,
      std::is_same<T, unsigned int>::value,
      std::is_same<T, long>::value,
      std::is_same<T, unsigned long>::value,
      std::is_same<T, float>::value,
#ifdef __STDCPP_BFLOAT16_T__
      std::is_same<T, half>::value,
      std::is_same<T, std::bfloat16_t>::value,
#endif
    };

  const auto result = std::any_of(
    std::begin(type_list),
    std::end(type_list),
    [](bool x)
    {
      return x;
    }
  );
  return false;
}

}

#endif //NAGATOLIB_METAL_NARRAY_IMPL_HPP_

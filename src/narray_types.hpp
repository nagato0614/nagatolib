//
// Created by toru on 2024/09/04.
//

#ifndef NAGATOLIB_SRC_NARRAY_TYPES_HPP_
#define NAGATOLIB_SRC_NARRAY_TYPES_HPP_

#include <vector>
#include <algorithm>

namespace nagato::na
{
// -----------------------------------------------------------------------------
// 前方宣言

// 外部からアクセスするためのクラス
template<typename T>
class NArray;

// 添字アクセス用
template<typename T>
class NArray_inner;


// -----------------------------------------------------------------------------
// using宣言
template<typename T>
using nVector = std::vector<T>;

template<typename T>
using SpanVector = std::span<T>;
}

// -----------------------------------------------------------------------------

template<typename T>
constexpr bool is_arithmetic_type()
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

  return result;
}

template<typename T>
constexpr bool requires_arithmetic_type = requires
{
  { is_arithmetic_type<T>() };
};

#endif //NAGATOLIB_SRC_NARRAY_TYPES_HPP_

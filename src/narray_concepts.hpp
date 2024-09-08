//
// Created by toru on 2024/09/07.
//

#ifndef NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_
#define NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_
#include <concepts>

namespace nagato::na
{

template<typename T>
concept nagato_arithmetic_c =
std::same_as<T, bool> ||
  std::same_as<T, char> ||
  std::same_as<T, unsigned char> ||
  std::same_as<T, int> ||
  std::same_as<T, unsigned int> ||
  std::same_as<T, long> ||
  std::same_as<T, unsigned long> ||
  std::same_as<T, float> ||
  std::same_as<T, size_t>;

// NagatoArray のコンセプト
template<typename ArrayType>
concept array_c =
requires(ArrayType array) {
  { array.Dimension_ };
  { array.Shapes_ };
  { array.TotalSize_ };
  { array[0] };
  { array.data };
  { array.begin() };
  { array.end() };
  std::same_as<decltype(array.begin()), std::add_pointer<typename ArrayType::ValueType>>;
  std::same_as<decltype(array.end()), std::add_pointer<typename ArrayType::ValueType>>;
};

// NagatoArray のサイズ比較用コンセプト
template<typename A, typename B>
concept array_size_c =
requires{
  { A::TotalSize_ };
};

// 関数として呼び出し可能かを確認する
template<typename F>
concept is_callable_one_c =
requires(F f)
{
  f(0);
};

template<typename F>
concept is_callable_two_c =
requires(F f)
{
  f(0, 0);
};

}

#endif //NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_

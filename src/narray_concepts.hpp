//
// Created by toru on 2024/09/07.
//

#ifndef NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_
#define NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_
#include <concepts>

namespace nagato::na
{

template<typename T>
concept NagatoArithmetic =
std::same_as<T, bool> ||
  std::same_as<T, char> ||
  std::same_as<T, unsigned char> ||
  std::same_as<T, int> ||
  std::same_as<T, unsigned int> ||
  std::same_as<T, long> ||
  std::same_as<T, unsigned long> ||
  std::same_as<T, float> ||
  std::same_as<T, size_t>;


template<typename ArrayType>
concept array_c = requires(ArrayType array) {
  { array.Dimension_ };
};

}

#endif //NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_

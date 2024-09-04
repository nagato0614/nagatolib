//
// Created by toru on 2024/09/04.
//

#ifndef NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_
#define NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_

#include <concepts>
#include "narray_types.hpp"


namespace nagato::na
{
// -----------------------------------------------------------------------------

// NArray_concept は NArray に必要なメンバーがあるかどうかを判定する
template<typename array_type, typename T> concept NArray_concept = requires {

  // メンバーに shape_ がある
  { array_type::shape_ } -> std::same_as<nVector<int>>;

  // メンバーに data_ は operator[] がある
  { array_type::data_[0] } -> std::same_as<T>;

  // メンバーに Shape() があり, nVector<int> を返す
  { array_type::Shape() } -> std::same_as<nVector<T>>;

  // メンバーに operator[] があり, NArray_inner を返す
  { array_type::operator[](0) } -> std::same_as<NArray_inner<T>>;

  // メンバーが算術型である
  { is_arithmetic_type<T>() };
};

}

#endif //NAGATOLIB_SRC_NARRAY_CONCEPTS_HPP_

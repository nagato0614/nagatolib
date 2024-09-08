//
// Created by toru on 2024/09/08.
//

#ifndef NAGATOLIB_SRC_NARRAY_FUNCTIONS_HPP_
#define NAGATOLIB_SRC_NARRAY_FUNCTIONS_HPP_
#include <iostream>
#include <span>
#include <iostream>
#include <array>
#include <numeric>
#include <cassert>

#include "narray.hpp"
#include "narray_concepts.hpp"

namespace nagato::na
{
// -----------------------------------------------------------------------------

/**
 * NagatoArray の同士の処理
 * 形状が同じ場合のみ処理を行う
 * 数値型が異なる場合は左辺値に合わせる
 * @tparam L
 * @tparam R
 * @param lhs
 * @param rhs
 * @return
 */
template<typename L, typename R, typename F>
auto Transform(const L &lhs, const R &rhs, F op)
-> typename L::template AsType<typename L::ValueType>
{
  // 左辺値が NagatoArrayFamily であることを確認
  static_assert(
    array_c<L>,
    "lhs is not NagatoArrayFamily"
  );

  // op が関数オブジェクトであることを確認
  static_assert(
    is_callable_c<F>,
    "op is not callable"
  );

  using ReturnType = typename L::template AsType<typename L::ValueType>;
  ReturnType result;

  // 両方が NagatoArrayFamily の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    // 形状が同じであることを確認
    static_assert(
      L::TotalSize_ == R::TotalSize_,
      "Shape is not match"
    );

    for (std::size_t i = 0; i < lhs.TotalSize_; ++i)
    {
      result.data[i] = static_cast<L::ValueType>(op(lhs.data[i], rhs.data[i]));
    }
  }
  else if (array_c<L> && !array_c<R>)
  {
    for (std::size_t i = 0; i < lhs.TotalSize_; ++i)
    {
      result.data[i] = static_cast<L::ValueType>(op(lhs.data[i], rhs));
    }
  }

  return result;
}

}

#endif //NAGATOLIB_SRC_NARRAY_FUNCTIONS_HPP_

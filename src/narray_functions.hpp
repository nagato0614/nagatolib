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
-> typename L::template AsType<L::ValueType>
{
  // 左辺値が NagatoArrayFamily であることを確認
  static_assert(
    array_c<L>,
    "lhs is not NagatoArrayFamily"
  );

  // 右辺値が NagatoArrayFamily であることを確認
  static_assert(
    array_c<R>,
    "rhs is not NagatoArrayFamily"
  );

  // opが関数であることを確認
  static_assert(
    std::is_function_v<F>,
    "op is not function"
  );

  // 形状が同じであることを確認
  static_assert(
    lhs.TotalSize_ == rhs.TotalSize_,
    "Shape is not match"
  );

  using ReturnType = L::template AsType<L::ValueType>;
  ReturnType result;

  return result;
}

}

#endif //NAGATOLIB_SRC_NARRAY_FUNCTIONS_HPP_

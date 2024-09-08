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
-> typename L::template AsType<typename L::ValueType>;

// -----------------------------------------------------------------------------

template<typename L, typename F>
auto Transform(const L &lhs, F op)
-> typename L::template AsType<typename L::ValueType>
{
  // op が関数オブジェクトであることを確認
  static_assert(
    is_callable_one_c<F>,
    "op is not callable"
  );

  static_assert
    (
      array_c<L>,
      "L is not NagatoArray"
    );

  using ReturnType = typename L::template AsType<typename L::ValueType>;
  ReturnType result;

  for (std::size_t i = 0; i < L::TotalSize_; ++i)
  {
    result.data[i] = static_cast<L::ValueType>(op(lhs.data[i]));
  }

  return result;
}

// -----------------------------------------------------------------------------

/**
 * NagatoArray 型を変える
 * @tparam ConvertType
 * @tparam ArrayType
 * @param array
 * @return
 */
template<typename ConvertType, typename ArrayType>
auto AsType(const ArrayType &array)
-> typename ArrayType::template AsType<ConvertType>;

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を表示する
 * @tparam ArrayType
 * @param array
 */
template<typename ArrayType>
void Show(const ArrayType &array);

// -----------------------------------------------------------------------------

/**
 * 指定した値で NagatoArray を初期化する
 * @tparam T
 * @tparam N
 * @param value
 * @return
 */
template<typename T, std::size_t... N>
NagatoArray<T, N...> Fill(T value);

// -----------------------------------------------------------------------------

/**
 * すべてを0で初期化する
 * @tparam T
 * @tparam N
 * @return
 */
template<typename T, std::size_t N>
NagatoArray<T, N> Zeros();

// -----------------------------------------------------------------------------

/**
 * NagatoArray のコピーを作成する
 * @tparam ArrayType
 * @param array
 * @return
 */
template<typename ArrayType>
auto Copy(const ArrayType &array);

// -----------------------------------------------------------------------------

}

#include "narray_functions_impl.hpp"
#endif //NAGATOLIB_SRC_NARRAY_FUNCTIONS_HPP_

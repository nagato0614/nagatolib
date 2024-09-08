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
#include <algorithm>

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
-> typename L::template AsType<typename L::ValueType>;

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

/**
 * NagatoArray の要素を足し算する
 * @tparam Primitive
 * @tparam size
 * @param a
 * @param b
 * @return
 */
template<typename L, typename R>
auto operator+(const L &lhs, const R &rhs);

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を引き算する
 * @tparam Primitive
 * @tparam size
 * @param a
 * @param b
 * @return
 */
template<typename L, typename R>
auto operator-(const L &lhs, const R &rhs);

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を掛け算する
 * @tparam Primitive
 * @tparam size
 * @param a
 * @param b
 * @return
 */
template<typename L, typename R>
auto operator*(const L &lhs, const R &rhs);

// -----------------------------------------------------------------------------
/**
 * NagatoArray の要素を割り算する
 * @tparam Primitive
 * @tparam size
 * @param a
 * @param b
 * @return
 */
template<typename L, typename R>
auto operator/(const L &lhs, const R &rhs);

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を比較する
 * @tparam L
 * @tparam R
 * @param lhs
 * @param rhs
 * @return
 */
template<typename L, typename R>
bool operator==(const L &lhs, const R &rhs);

// -----------------------------------------------------------------------------

template<typename L, typename R>
bool operator!=(const L &lhs, const R &rhs)
{
  return !(lhs == rhs);
}

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を比較する
 * @tparam L
 * @tparam R
 * @param lhs
 * @param rhs
 * @return
 */
template<typename L, typename R>
auto operator<(const L &lhs, const R &rhs)
-> typename L::template AsType<bool>
{
  using ResultType = typename L::template AsType<bool>;
  ResultType result;
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    for (std::size_t i = 0; i < lhs.TotalSize_; ++i)
    {
      result.data[i] = lhs.data[i] < rhs.data[i];
    }
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return Transform(lhs, [rhs](auto a)
    { return a < rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return lhs < a; });
  }

  return result;
}

// -----------------------------------------------------------------------------
/**
 * NagatoArray の要素を比較する
 * @tparam L
 * @tparam R
 * @param lhs
 * @param rhs
 * @return
 */
template<typename L, typename R>
auto operator>(const L &lhs, const R &rhs)
-> typename L::template AsType<bool>
{
  using ResultType = typename L::template AsType<bool>;
  ResultType result;
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    for (std::size_t i = 0; i < lhs.TotalSize_; ++i)
    {
      result.data[i] = lhs.data[i] > rhs.data[i];
    }
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return Transform(lhs, [rhs](auto a)
    { return a > rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return lhs > a; });
  }

  return result;
}

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を比較する
 * @tparam L
 * @tparam R
 * @param lhs
 * @param rhs
 * @return
 */
template<typename L, typename R>
auto operator<=(const L &lhs, const R &rhs)
-> typename L::template AsType<bool>
{
  using ResultType = typename L::template AsType<bool>;
  ResultType result;
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    for (std::size_t i = 0; i < lhs.TotalSize_; ++i)
    {
      result.data[i] = lhs.data[i] <= rhs.data[i];
    }
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return Transform(lhs, [rhs](auto a)
    { return a <= rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return lhs <= a; });
  }

  return result;
}

// -----------------------------------------------------------------------------

/**
 * NagatoArray の要素を比較する
 * @tparam L
 * @tparam R
 * @param lhs
 * @param rhs
 * @return
 */
template<typename L, typename R>
auto operator>=(const L &lhs, const R &rhs)
-> typename L::template AsType<bool>
{
  using ResultType = typename L::template AsType<bool>;
  ResultType result;
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    for (std::size_t i = 0; i < lhs.TotalSize_; ++i)
    {
      result.data[i] = lhs.data[i] >= rhs.data[i];
    }
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return Transform(lhs, [rhs](auto a)
    { return a >= rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return lhs >= a; });
  }

  return result;
}


} // namespace nagato::na

#include "narray_functions_impl.hpp"
#endif //NAGATOLIB_SRC_NARRAY_FUNCTIONS_HPP_

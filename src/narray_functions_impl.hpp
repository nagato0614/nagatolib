//
// Created by toru on 2024/09/08.
//

#ifndef NAGATOLIB_SRC_NARRAY_FUNCTIONS_IMPL_HPP_
#define NAGATOLIB_SRC_NARRAY_FUNCTIONS_IMPL_HPP_

#include "narray.hpp"
#include "narray_concepts.hpp"
#include "narray_functions.hpp"

namespace nagato::na
{

template<typename L, typename R, typename F>
auto Transform(const L &lhs,
               const R &rhs,
               F op) -> typename L::template AsType<typename L::ValueType>
{
  // op が関数オブジェクトであることを確認
  static_assert(
    is_callable_two_c<F>,
    "op is not callable"
  );

  static_assert(
    array_c<L>,
    "L is not NagatoArray"
  );

  static_assert(
    array_c<R>,
    "R is not NagatoArray"
  );

  using ReturnType = typename L::template AsType<typename L::ValueType>;
  ReturnType result;

  // 形状が同じであることを確認
  static_assert(
    L::TotalSize_ == R::TotalSize_,
    "Shape is not match"
  );

  // 各shape が同じであることを確認
  for (std::size_t i = 0; i < L::Shapes_.size(); ++i)
  {
    assert(L::Shapes_[i] == R::Shapes_[i]);
  }

  for (std::size_t i = 0; i < L::TotalSize_; ++i)
  {
    result.data[i] = static_cast<L::ValueType>(op(lhs.data[i], rhs.data[i]));
  }
  return result;
}

// -----------------------------------------------------------------------------

template<typename L, typename F>
auto Transform(const L &lhs, F op) -> typename L::template AsType<typename L::ValueType>
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

template<typename ConvertType, typename ArrayType>
auto AsType(const ArrayType &array)
-> typename ArrayType::template AsType<ConvertType>
{
  // ConvertType が算術型かどうかを判定する
  static_assert(nagato_arithmetic_c<ConvertType>);

  using ConvertArrayType = typename ArrayType::template AsType<ConvertType>;

  ConvertArrayType result;

  std::transform(
    array.begin(),
    array.end(),
    result.begin(),
    [](auto value) -> ConvertType
    {
      return static_cast<ConvertType>(value);
    }
  );

  return result;
}
// -----------------------------------------------------------------------------

template<typename ArrayType>
void Show_impl(const ArrayType &array)
{
  static_assert(
    array_c<ArrayType>,
    "Show function is only supported for NagatoArrayFamily"
  );

  if constexpr (ArrayType::Dimension_ == 1)
  {
    std::cout << "[";
    for (std::size_t i = 0; i < ArrayType::Shapes_[0]; i++)
    {
      if (std::same_as<typename ArrayType::ValueType, bool>)
      {
        std::cout << (array[i] ? "true" : "false");
      }
      else
      {
        std::cout << array[i];
      }
      if (i != ArrayType::Shapes_[0] - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]";
  }
  else
  {
    std::cout << "[";
    for (std::size_t i = 0; i < ArrayType::Shapes_[0]; i++)
    {
      Show_impl(array[i]);
      if (i != ArrayType::Shapes_[0] - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]";
  }
}

template<typename ArrayType>
void Show(const ArrayType &array)
{
  static_assert(
    array_c<ArrayType>,
    "Show function is only supported for NagatoArrayFamily"
  );

  Show_impl(array);
  std::cout << std::endl;
}

// -----------------------------------------------------------------------------


template<typename T, std::size_t N>
NagatoArray<T, N> Zeros()
{
  return Fill<T, N>(static_cast<T>(0));
}

// -----------------------------------------------------------------------------

template<typename ArrayType>
auto Copy(const ArrayType &array)
{
  static_assert(
    array_c<ArrayType>,
    "Copy function is only supported for NagatoArrayFamily"
  );

  typename ArrayType::CopyType copy(array);

  return copy;
}

// -----------------------------------------------------------------------------

template<typename T, size_t... N>
NagatoArray<T, N...> Fill(T value)
{
  // Tが数値演算可能な型であるかどうかを判定する
  static_assert(
    nagato_arithmetic_c<T>,
    "T is not arithmetic"
  );

  return NagatoArray<T, N...>(value);
}

// -----------------------------------------------------------------------------

template<typename L, typename R>
auto operator+(const L &lhs, const R &rhs)
{
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    return Transform(lhs, rhs, [](auto a, auto b)
    { return a + b; });
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return Transform(lhs, [rhs](auto a)
    { return a + rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return a + lhs; });
  }
}

// -----------------------------------------------------------------------------
template<typename L, typename R>
auto operator-(const L &lhs, const R &rhs)
{
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    return Transform(lhs, rhs, [](auto a, auto b)
    { return a - b; });
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return Transform(lhs, [rhs](auto a)
    { return a - rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return a - lhs; });
  }
}

// -----------------------------------------------------------------------------

template<typename L, typename R>
auto operator*(const L &lhs, const R &rhs)
{
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    return Transform(lhs, rhs, [](auto a, auto b)
    { return a * b; });
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return Transform(lhs, [rhs](auto a)
    { return a * rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return a * lhs; });
  }
}

// -----------------------------------------------------------------------------

template<typename L, typename R>
auto operator/(const L &lhs, const R &rhs)
{
  // 左辺値がArray型, 右辺値がArray型の場合
  if constexpr (array_c<L> && array_c<R>)
  {
    return Transform(lhs, rhs, [](auto a, auto b)
    { return a / b; });
  }
    // 左辺値がArray型, 右辺値が数値型の場合
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    if (rhs == 0)
    {
      throw std::runtime_error("Division by zero");
    }
    return Transform(lhs, [rhs](auto a)
    { return a / rhs; });
  }
    // 左辺値が数値型, 右辺値がArray型の場合
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return Transform(rhs, [lhs](auto a)
    { return lhs / a; });
  }
}

// -----------------------------------------------------------------------------

template<typename L, typename R>
bool operator==(const L &lhs, const R &rhs)
{

  if constexpr (array_c<L> && array_c<R>)
  {
    return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  }
  else if constexpr (array_c<L> && nagato_arithmetic_c<R>)
  {
    return std::all_of(lhs.begin(), lhs.end(), [rhs](auto a)
    { return a == rhs; });
  }
  else if constexpr (nagato_arithmetic_c<L> && array_c<R>)
  {
    return std::all_of(rhs.begin(), rhs.end(), [lhs](auto a)
    { return a == lhs; });
  }
}
// -----------------------------------------------------------------------------

} // namespace nagato::na

#endif //NAGATOLIB_SRC_NARRAY_FUNCTIONS_IMPL_HPP_

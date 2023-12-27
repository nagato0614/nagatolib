//
// Created by nagato0614 on 2019-07-14.
//

#ifndef NAGATOLIB_SRC_MATRIX_HPP_
#define NAGATOLIB_SRC_MATRIX_HPP_

#include <cstdlib>
#include <iostream>

#include "assert.hpp"
#include "vector.hpp"
#include "math.hpp"

namespace nagato
{

// -----------------------------------------------------------------------------

/**
 * 固定長のMatrixクラスVectoraクラスと互換性を持つ
 * @tparam Primitive
 * @tparam Column 列 : 縦
 * @tparam Row 行 : 横
 */
template<typename Primitive, std::size_t Column, std::size_t Row>
class Matrix
{
  STATIC_ASSERT_IS_ARITHMETRIC(Primitive);

 public:
  using self = Matrix<Primitive, Column, Row>;
  using row_array = Vector<Primitive, Row>;
  using matrix = std::array<row_array, Column>;
  using size = std::size_t;
  using reference = self &;
  using const_reference = const self &;
  using rvalue_reference = self &;

  constexpr explicit
  Matrix(Primitive p = 0.0)
  noexcept
  {
    for (auto &row : matrix_)
      row.Fill(p);
  }

  /**
   * 初期化リストからMatrixを生成する
   * @param init
   */
  constexpr
  Matrix(const std::initializer_list<std::initializer_list<Primitive>> &init)
  noexcept
  {
    assert(init.size() <= Column);
    for (size i = 0; i < init.size(); i++)
    {
      const auto &r = *(init.begin() + i);
      assert(r.size() <= Row);
      matrix_[i] = r;
    }
  }

  constexpr
  Matrix(const std::initializer_list<Primitive> &init)
  noexcept
  {
    assert(init.size() <= Column * Row);
    size i = 0;
    for (const auto &r : init)
    {
      matrix_[i / Row][i % Row] = r;
      i++;
    }
  }

  constexpr
  Matrix(const_reference m) noexcept
    : matrix_(m.matrix_)
  {}

  constexpr
  Matrix(rvalue_reference m) noexcept
    : matrix_(m.matrix_)
  {}

  ~Matrix() = default;

  constexpr static
  self Zero() noexcept
  {
    return Matrix<Primitive, Row, Column>(0.0);
  }

  constexpr static
  self Identity() noexcept
  {
    Matrix<Primitive, Row, Column> m(0);
    m.FillDiagonal(1);
    return m;
  }

  constexpr reference operator=(const_reference v) noexcept
  {
    matrix_ = v.matrix_;
    return *this;
  }

  constexpr row_array &operator[](size index)
  & noexcept
  {
    assert(0 <= index && index < Column);
    return matrix_[index];
  }

  constexpr const row_array &operator[](size index)
  const & noexcept
  {
    assert(0 <= index && index < Column);
    return matrix_[index];
  }

  constexpr row_array operator[](size index)
  const && noexcept
  {
    assert(0 <= index && index < Column);
    return matrix_[index];
  }

  constexpr reference operator+=(const_reference v) noexcept
  {
    for (int i = 0; i < Row; i++)
      matrix_[i] += v[i];
    return *this;
  }

  template<typename T>
  constexpr self &operator+=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Row; i++)
      matrix_[i] += value;
    return *this;
  }

  constexpr reference operator-=(const_reference v) noexcept
  {
    for (int i = 0; i < Row; i++)
      matrix_[i] -= v[i];
    return *this;
  }

  template<typename T>
  constexpr self &operator-=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Row; i++)
      matrix_[i] -= value;
    return *this;
  }

  constexpr reference operator*=(const_reference v) noexcept
  {
    matrix result;

    for (size row = 0; row < Row; row++)
    {
      for (size column = 0; column < Column; column++)
      {
        result[row][column] = matrix_[row][column] * v.matrix_[row][column];
      }
    }

    matrix_ = result;
    return *this;
  }

  constexpr reference operator/=(const_reference v) noexcept
  {
    matrix result;

    for (size row = 0; row < Row; row++)
    {
      for (size column = 0; column < Column; column++)
      {
        result[row][column] = matrix_[row][column] / v.matrix_[row][column];
      }
    }

    matrix_ = result;
    return *this;
  }

  template<typename T>
  constexpr self &operator*=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Row; i++)
      matrix_[i] *= value;
    return *this;
  }

  template<typename T>
  constexpr self &operator/=(const T &value) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (int i = 0; i < Row; i++)
      matrix_[i] /= value;
    return *this;
  }

  constexpr void Sqrt() noexcept
  {
    for (size i = 0; i < Row; i++)
      matrix_[i].Sqrt();
  }

  constexpr bool HasNaN() const noexcept
  {
    for (size i = 0; i < Row; i++)
      if (matrix_[i].HasNan())
      {
        return true;
      }
    return false;
  }

  constexpr bool HasZero() const noexcept
  {
    for (size i = 0; i < Row; i++)
      if (matrix_[i].HasZero())
      {
        return true;
      }
    return false;
  }

  constexpr auto begin() const noexcept
  {
    return matrix_.begin();
  }

  constexpr auto end() const noexcept
  {
    return matrix_.end();
  }

  constexpr Primitive Max() const noexcept
  {
    Primitive m = 0;
    for (const auto &i : matrix_)
      m = max(m, i.Max());
    return m;
  }

  constexpr Primitive Min() const noexcept
  {
    Primitive m = 0;
    for (const auto &i : matrix_)
      m = min(m, i.Min());
    return m;
  }

  template<typename T>
  constexpr void Fill(T t) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    for (auto &i : matrix_)
      i.Fill(t);
  }

  template<typename T>
  constexpr void FillDiagonal(const T &t) noexcept
  {
    STATIC_ASSERT_IS_ARITHMETRIC(T);
    size diag = Row < Column ? Row : Column;
    for (size i = 0; i < diag; i++)
      matrix_[i][i] = t;
  }

  template<typename L, typename R>
  constexpr void Clamp(L l, R r) noexcept
  {
    for (auto &i : matrix_)
      i = Clamp(i, l, r);
  }

  constexpr Primitive Sum()
  const noexcept
  {
    Primitive sum = 0.0;
    for (const auto &i : matrix_)
      sum += i;
    return sum;
  }

  constexpr std::pair<Primitive, Primitive> Shape()
  const noexcept
  {
    return std::make_pair(Column, Row);
  }

  constexpr void ShowPair() const noexcept
  {
    std::cout << "(" << Column << ", " << Row << ")" << std::endl;
  }

 private:

  matrix matrix_;
};
// -----------------------------------------------------------------------------

template<class Char, class Traits, class Primitive, std::size_t Row,
  std::size_t Column>
std::basic_ostream<Char, Traits> &operator<<(
  std::basic_ostream<Char, Traits> &os,
  const Matrix<Primitive, Column, Row> &dt)
{
  os << "[";
  for (std::size_t row = 0; row < Row; row++)
  {
    if (row != 0)
    {
      os << " ";
    }
    os << "[";
    for (std::size_t column = 0; column < Column; column++)
    {
      os << dt[column][row];
      if (column != Column - 1)
      {
        os << ", ";
      }
    }
    os << "]";
    if (row != Row - 1)
    {
      os << "\n";
    }
  }
  os << "]";
  return os;
}

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t Row, std::size_t Column>
constexpr Matrix<Primitive, Row, Column> operator+(
  const Matrix<Primitive, Row, Column> &lhv,
  const Matrix<Primitive, Row, Column> &rhv
) noexcept
{
  return Matrix(lhv) += rhv;
}

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t Row, std::size_t Column>
constexpr Matrix<Primitive, Row, Column> operator-(
  const Matrix<Primitive, Row, Column> &lhv,
  const Matrix<Primitive, Row, Column> &rhv
) noexcept
{
  return Matrix(lhv) -= rhv;
}

// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t Row, std::size_t Column>
constexpr Matrix<Primitive, Row, Column> operator*(
  const Matrix<Primitive, Row, Column> &lhv,
  const Matrix<Primitive, Row, Column> &rhv
) noexcept
{
  return Matrix(lhv) *= rhv;
}

template<typename Primitive, std::size_t Row, std::size_t Column,
  std::size_t Size>
constexpr Vector<Primitive, Size> Dot(
  const Matrix<Primitive, Row, Column> &lhv,
  const Vector<Primitive, Size> &rhv
) noexcept
{
  static_assert(Size == Column);

  Vector<Primitive, Size> result(0);
  for (std::size_t i = 0; i < Size; i++)
  {
    Primitive sum = 0;
    for (std::size_t j = 0; j < Size; j++)
    {
      sum += lhv[i][j] * rhv[j];
    }
    result[i] = sum;
  }
  return result;
}

template<typename Primitive, std::size_t Row, std::size_t Column,
  std::size_t Size>
constexpr Vector<Primitive, Size> Dot(
  const Vector<Primitive, Size> &lhv,
  const Matrix<Primitive, Row, Column> &rhv
) noexcept
{
  static_assert(Size == Row);

  Vector<Primitive, Size> result(0);
  for (std::size_t i = 0; i < Size; i++)
  {
    Primitive sum = 0;
    for (std::size_t j = 0; j < Size; j++)
      sum += lhv[j] * rhv[j][i];
    result[i] = sum;
  }
  return result;
}

template<typename Primitive,
  std::size_t m1_Row, std::size_t m1_Column,
  std::size_t m2_Row, std::size_t m2_Column>
constexpr Matrix<Primitive, m1_Column, m2_Row> Dot(
  const Matrix<Primitive, m1_Column, m1_Row> &lhv,
  const Matrix<Primitive, m2_Column, m2_Row> &rhv
) noexcept
{
  static_assert(m1_Row == m2_Column);

  Matrix<Primitive, m1_Column, m2_Row> result(0);
  for (std::size_t i = 0; i < m1_Column; i++)
  {
    for (std::size_t j = 0; j < m2_Row; j++)
    {
      Primitive sum = 0;
      for (std::size_t k = 0; k < m1_Row; k++)
        sum += lhv[i][k] * rhv[k][j];
      result[i][j] = sum;
    }
  }
  return result;
}

/**
 * サイズが 1 x n の行列と 1 x n の場合は内積を計算する
 * @tparam Primitive
 * @tparam Column
 * @param lhv
 * @param rhv
 * @return
 */
template<typename Primitive, std::size_t Column>
constexpr Primitive Dot(
  const Matrix<Primitive, 1, Column> &lhv,
  const Matrix<Primitive, 1, Column> &rhv
) noexcept
{
  Primitive result = 0;
  for (std::size_t i = 0; i < Column; i++)
    result += lhv[0][i] * rhv[0][i];
  return result;
}


// -----------------------------------------------------------------------------

template<typename Primitive, std::size_t Row, std::size_t Column>
constexpr bool operator==(
  const Matrix<Primitive, Row, Column> &lhv,
  const Matrix<Primitive, Row, Column> &rhv
) noexcept
{
  bool result = true;
  for (std::size_t i = 0; i < Row; i++)
  {
    result &= lhv[i] == rhv[i];
  }
  return result;
}

}

#endif //NAGATOLIB_SRC_MATRIX_HPP_

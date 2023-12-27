//
// Created by toru on 2023/12/27.
//

#ifndef NAGATOLIB_SRC_MATRIX_N_HPP_
#define NAGATOLIB_SRC_MATRIX_N_HPP_
#include "assert.hpp"
#include "type_traits.hpp"
#include <iostream>
#include <vector>

namespace nagato
{
/**
 * @brief　可変長の行列を表すクラス
 */
template<typename Primitive>
class MatrixN
{
  STATIC_ASSERT_IS_ARITHMETRIC(Primitive);
 public:
  using self = MatrixN<Primitive>;
  using row_array = std::vector<Primitive>;
  using matrix_array = std::vector<row_array>;
  using reference = self &;
  using const_reference = const self &;
  using rvalue_reference = self &&;

  /**
   * @brief コンストラクタ
   * @param Column 列数
   * @param Row 行数
   */
  MatrixN(std::size_t Row, std::size_t Column)
    : column_(Column), row_(Row), matrix_(Row, row_array(Column))
  {
  }

  /**
   * @brief コンストラクタ
   * @param list
   */
  MatrixN(const std::initializer_list<std::initializer_list<Primitive>> &list)
    : column_(list.begin()->size()), row_(list.size()), matrix_(row_, row_array(column_))
  {
    for (std::size_t i = 0; i < row_; i++)
    {
      const auto &row_list = *(list.begin() + i);
      for (std::size_t j = 0; j < column_; j++)
      {
        const auto &element = *(row_list.begin() + j);
        matrix_[i][j] = element;
      }
    }
  }

  /**
   * @brief コンストラクタ
   * ベクトルとして取り扱う
   * @param list
   */
  MatrixN(const std::initializer_list<Primitive> &list)
    : column_(list.size()), row_(1), matrix_(row_, row_array(column_))
  {
    std::size_t i = 0;
    for (const auto &element : list)
    {
      matrix_[0][i] = element;
      i++;
    }
  }

  /**
   * @brief コピーコンストラクタ
   * @param other
   */
  MatrixN(const self &other)
    : column_(other.column_), row_(other.row_), matrix_(other.matrix_)
  {}

  /**
   * @brief ムーブコンストラクタ
   * @param other
   */
  MatrixN(self &&other) noexcept
    : column_(other.column_), row_(other.row_), matrix_(std::move(other.matrix_))
  {}

  /**
   * @brief 行列のサイズが等しいことを確認する
   */
  void assert_same_size(const self &other) const
  {
    assert(column_ == other.column_);
    assert(row_ == other.row_);
  }

  /**
   * @brief コピー代入演算子
   * @param other
   * @return
   */
  reference operator=(const self &other)
  {
    column_ = other.column_;
    row_ = other.row_;
    matrix_ = other.matrix_;
    return *this;
  }

  /**
   * @brief ムーブ代入演算子
   * @param other
   * @return
   */
  reference operator=(self &&other) noexcept
  {
    column_ = other.column_;
    row_ = other.row_;
    matrix_ = std::move(other.matrix_);
    return *this;
  }

  /**
   * @brief 行列の要素を取得する
   * @param column
   * @return
   */
  row_array &operator[](std::size_t column)
  {
    return matrix_[column];
  }

  /**
   * @brief 行列の要素を取得する
   * @param row
   * @return
   */
  const row_array &operator[](std::size_t row) const
  {
    return matrix_[row];
  }

  /**
   * @brief 行列の要素を取得する
   * @param column
   * @param row
   * @return
   */
  Primitive &at(std::size_t column, std::size_t row)
  {
    return matrix_.at(column).at(row);
  }

  /**
   * @brief 行列の要素を取得する
   * @param column
   * @param row
   * @return
   */
  const Primitive &at(std::size_t column, std::size_t row) const
  {
    return matrix_.at(column).at(row);
  }

  /**
   * @brief 要素ごとの加算
   * @param column
   * @param row
   * @return
   */
  self operator+(const self &other) const
  {
    this->assert_same_size(other);
    self matrix(column_, row_);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix[i][j] = matrix_[i][j] + other[i][j];
      }
    }
    return matrix;
  }

  /**
   * @brief 要素ごとの加算代入
   * @param column
   * @param row
   * @return
   */
  reference operator+=(const self &other)
  {
    this->assert_same_size(other);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix_[i][j] += other[i][j];
      }
    }
    return *this;
  }

  /**
   * @brief 要素ごとの減算
   * @param column
   * @param row
   * @return
   */
  self operator-(const self &other) const
  {
    this->assert_same_size(other);
    self matrix(column_, row_);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix[i][j] = matrix_[i][j] - other[i][j];
      }
    }
    return matrix;
  }

  /**
   * @brief 要素ごとの減算代入
   * @param column
   * @param row
   * @return
   */
  reference operator-=(const self &other)
  {
    this->assert_same_size(other);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] -= other[i][j];
      }
    }
    return *this;
  }

  /**
   * @brief 要素ごとの乗算
   * @param column
   * @param row
   * @return
   */
  self operator*(const self &other) const
  {
    this->assert_same_size(other);
    self matrix(column_, row_);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix[i][j] = matrix_[i][j] * other[i][j];
      }
    }
    return matrix;
  }

  /**
   * @brief 要素ごとの乗算代入
   * @param column
   * @param row
   * @return
   */
  reference operator*=(const self &other)
  {
    this->assert_same_size(other);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] *= other[i][j];
      }
    }
    return *this;
  }

  /**
   * @brief 要素ごとの除算
   * @param column
   * @param row
   * @return
   */
  self operator/(const self &other) const
  {
    this->assert_same_size(other);
    self matrix(column_, row_);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix[i][j] = matrix_[i][j] / other[i][j];
      }
    }
    return matrix;
  }

  /**
   * @brief 要素ごとの除算代入
   * @param column
   * @param row
   * @return
   */
  reference operator/=(const self &other)
  {
    this->assert_same_size(other);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] /= other[i][j];
      }
    }
    return *this;
  }

  static self Zero(std::size_t Column, std::size_t Row)
  {
    return self(Column, Row);
  }

  static self Identity(std::size_t Column, std::size_t Row)
  {
    self matrix(Column, Row);
    for (std::size_t i = 0; i < Column; i++)
    {
      for (std::size_t j = 0; j < Row; j++)
      {
        if (i == j)
        {
          matrix[i][j] = 1;
        }
        else
        {
          matrix[i][j] = 0;
        }
      }
    }
    return matrix;
  }

  std::size_t Column() const
  {
    return column_;
  }

  std::size_t Row() const
  {
    return row_;
  }

  /**
   * @brief 行列の列数を取得する
   * @return
   */
  void ShowShape() const
  {
    std::cout << "Column: " << column_ << ", Row: " << row_ << std::endl;
  }

  void Sqrt()
  {
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] = std::sqrt(matrix_[i][j]);
      }
    }
  }

  bool HasNaN() const
  {
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        if (std::isnan(matrix_[i][j]))
        {
          return true;
        }
      }
    }
    return false;
  }

  bool HasZero() const
  {
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        if (matrix_[i][j] == 0)
        {
          return true;
        }
      }
    }
    return false;
  }

  auto begin() const
  {
    return matrix_.begin();
  }

  auto end() const
  {
    return matrix_.end();
  }

  Primitive Sum() const
  {
    Primitive sum = 0;
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        sum += matrix_[i][j];
      }
    }
    return sum;
  }

  Primitive Max() const
  {
    Primitive max = matrix_[0][0];
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        if (max < matrix_[i][j])
        {
          max = matrix_[i][j];
        }
      }
    }
    return max;
  }

  Primitive Min() const
  {
    Primitive min = matrix_[0][0];
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        if (min > matrix_[i][j])
        {
          min = matrix_[i][j];
        }
      }
    }
    return min;
  }

  template<typename T>
  void Fill(T t)
  {
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] = static_cast<Primitive>(t);
      }
    }
  }

  template<typename F>
  void itor(F &&f)
  {
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] = f(matrix_[i][j]);
      }
    }
  }

 private:
  std::size_t column_;
  std::size_t row_;
  matrix_array matrix_;
};

template<class Char, class Traits, typename Primitive>
std::basic_ostream<Char, Traits> &operator<<(std::basic_ostream<Char, Traits> &os,
                                             const MatrixN<Primitive> &matrix)
{
  os << "[";
  for (std::size_t i = 0; i < matrix.column_; i++)
  {
    if (i != 0)
    {
      os << " ";
    }
    os << "[";
    for (std::size_t j = 0; j < matrix.row_; j++)
    {
      os << matrix[i][j] << ", ";
    }
    os << "]";
    if (i != matrix.column_ - 1)
    {
      os << std::endl;
    }
    os << std::endl;
  }
  os << "]";
  return os;
}

template<typename Primitive>
MatrixN<Primitive> Dot(const MatrixN<Primitive> &a, const MatrixN<Primitive> &b)
{
  bool is_matrix = (a.Column() == b.Row());
  bool is_vector = ((a.Row() == 1) && (b.Row() == 1) && (a.Column() == b.Column()))
    || ((a.Column() == 1) && (b.Column() == 1) && (a.Row() == b.Row()));
  // 行列の積を計算する
  if (is_matrix)
  {
    MatrixN<Primitive> matrix(a.Row(), b.Column());
    for (std::size_t i = 0; i < a.Row(); i++)
    {
      for (std::size_t j = 0; j < b.Column(); j++)
      {
        for (std::size_t k = 0; k < a.Column(); k++)
        {
          matrix[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return matrix;
  }
    // ベクトルの内積を計算する
  else if (is_vector)
  {
    MatrixN<Primitive> matrix(1, 1);
    for (std::size_t i = 0; i < a.Column(); i++)
    {
      matrix[0][0] += a[0][i] * b[0][i];
    }
    return matrix;
  }
  else
  {
    throw std::invalid_argument("The size of the matrix is invalid.");
  }
}

} // namespace nagato

#endif //NAGATOLIB_SRC_MATRIX_N_HPP_

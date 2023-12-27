//
// Created by toru on 2023/12/27.
//

#ifndef NAGATOLIB_SRC_MATRIX_N_HPP_
#define NAGATOLIB_SRC_MATRIX_N_HPP_
#include "assert.hpp"
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
  MatrixN(std::size_t Column, std::size_t Row)
    : column_(Column), row_(Row), matrix_(Column, row_array(Row))
  {
  }

  /**
   * @brief コンストラクタ
   * @param list
   */
  MatrixN(const std::initializer_list<std::initializer_list<Primitive>> &list)
    : column_(list.size()), row_(list.begin()->size()), matrix_(column_, row_array(row_))
  {
    std::size_t i = 0;
    for (const auto &row : list)
    {
      std::size_t j = 0;
      for (const auto &element : row)
      {
        matrix_[i][j] = element;
        j++;
      }
      i++;
    }
  }

  /**
   * @brief コンストラクタ
   * ベクトルとして取り扱う
   * @param list
   */
  MatrixN(const std::initializer_list<Primitive> &list)
    : column_(1), row_(list.size()), matrix_(column_, row_array(row_))
  {
    std::size_t i = 0;
    for (const auto &element : list)
    {
      matrix_[i][0] = element;
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
   * @param column
   * @return
   */
  const row_array &operator[](std::size_t column) const
  {
    return matrix_[column];
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
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
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

 private:
  std::size_t column_;
  std::size_t row_;
  matrix_array matrix_;
};
}

#endif //NAGATOLIB_SRC_MATRIX_N_HPP_

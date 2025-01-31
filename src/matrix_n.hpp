//
// Created by toru on 2023/12/27.
//

#ifndef NAGATOLIB_SRC_MATRIX_N_HPP_
#define NAGATOLIB_SRC_MATRIX_N_HPP_
#include "assert.hpp"
#include "type_traits.hpp"
#include "random.hpp"
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
   */
  MatrixN() = default;

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
   * @brief コンストラクタ
   * @param list
   */
  MatrixN(const std::vector<std::vector<Primitive>> &list)
    : column_(list.begin()->size()), row_(list.size())
  {
    matrix_.resize(row_);
    for (std::size_t i = 0; i < row_; i++)
    {
      matrix_[i].resize(column_);
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix_[i][j] = list[i][j];
      }
    }
  }

  /**
   * @brief コンストラクタ
   * ベクトルとして取り扱う
   * @param list
   */
  MatrixN(const std::vector<Primitive> &list)
    : column_(list.size()), row_(1)
  {
    matrix_.resize(row_);
    matrix_[0].resize(column_);
    for (std::size_t i = 0; i < column_; i++)
    {
      matrix_[0][i] = list[i];
    }
  }

  /**
   * @brief 1次元のベクトルを2次元の行列に変換する
   * @param list
   * @param row
   * @param column
   * @return
   */
  MatrixN(const std::vector<Primitive> &list, std::size_t row, std::size_t column)
    : column_(column), row_(row)
  {
    assert(list.size() == row * column);
    matrix_.resize(row);
    for (std::size_t i = 0; i < row; i++)
    {
      matrix_[i].resize(column);
      for (std::size_t j = 0; j < column; j++)
      {
        matrix_[i][j] = list[i * column + j];
      }
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
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
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
   * スカラーとの加算
   * @param other
   * @return
   */
  self operator+(const Primitive &other) const
  {
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix[i][j] = matrix_[i][j] + static_cast<Primitive>(other);
      }
    }
    return matrix;
  }

  /**
   * スカラーとの加算代入
   * @param other
   * @return
   */
  reference operator+=(const Primitive &other)
  {
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix_[i][j] += static_cast<Primitive>(other);
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
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
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
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix_[i][j] -= other[i][j];
      }
    }
    return *this;
  }

  /**
   * スカラーとの減算
   * @param other
   * @return
   */
  self operator-(const Primitive &other) const
  {
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix[i][j] = matrix_[i][j] - static_cast<Primitive>(other);
      }
    }
    return matrix;
  }

  /**
   * スカラーとの減算代入
   * @param other
   * @return
   */
  reference operator-=(const Primitive &other)
  {
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix_[i][j] -= static_cast<Primitive>(other);
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
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
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
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix_[i][j] *= other[i][j];
      }
    }
    return *this;
  }

  /**
   * @brief スカラーとの乗算
   * @param other
   * @return
   */
  self operator*(const Primitive &other) const
  {
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix[i][j] = matrix_[i][j] * static_cast<Primitive>(other);
      }
    }
    return matrix;
  }

  /**
   * @brief スカラーとの乗算代入
   * @param other
   * @return
   */
  reference operator*=(const Primitive &other)
  {
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix_[i][j] *= static_cast<Primitive>(other);
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
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
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

  /**
   * @brief スカラーとの除算
   * @param other
   * @return
   */
  self operator/(const Primitive &other) const
  {
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix[i][j] = matrix_[i][j] / static_cast<Primitive>(other);
      }
    }
    return matrix;
  }

  /**
   * @brief スカラーとの除算代入
   * @param other
   * @return
   */
  reference operator/=(const Primitive &other)
  {
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] /= static_cast<Primitive>(other);
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

  self Exp()
  {
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix[i][j] = std::exp(matrix_[i][j]);
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
    std::cout << "Row: " << row_ << ", Column: " << column_ << std::endl;
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
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        sum += matrix_[i][j];
      }
    }
    return sum;
  }

  Primitive Max() const
  {
    Primitive max = matrix_[0][0];
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
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
  void Itor(F &&f)
  {
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix_[i][j] = f(matrix_[i][j]);
      }
    }
  }

  template<typename F>
  auto Itor(F &&f) const
  {
    self matrix(row_, column_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        matrix[i][j] = f(matrix_[i][j]);
      }
    }
    return matrix;
  }

  /**
   * @brief 0.0 ~ 1.0 の間で正規化を行う
   */
  void Normalize()
  {
    const Primitive max = Max();
    const Primitive min = Min();
    const Primitive diff = max - min;
    const auto one = static_cast<Primitive>(1);
    const auto zero = static_cast<Primitive>(0);

    // 全ての要素が同じ値の場合は0で埋める
    if (diff == zero)
    {
      Fill(zero);
    }
      // それ以外の場合は正規化を行う
    else
    {
      Itor([diff, min, one](Primitive x) -> Primitive
           { return (x - min) / diff; });
    }
  }

  /**
   * @brief 0.0 ~ 1.0 の間で正規化を行う
   */
  self Normalized() const
  {
    const Primitive max = Max();
    const Primitive min = Min();
    const Primitive diff = max - min;
    const auto one = static_cast<Primitive>(1);
    const auto zero = static_cast<Primitive>(0);

    // 全ての要素が同じ値の場合は0で埋める
    if (diff == zero)
    {
      return Zero(column_, row_);
    }
      // それ以外の場合は正規化を行う
    else
    {
      return Itor([diff, min, one](Primitive x) -> Primitive
                  { return (x - min) / diff; });
    }
  }

  /**
   * @brief 2次元の行列を1次元のベクトル形式にする
   */
  void ToVector()
  {
    row_array vector(column_ * row_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        vector[i * column_ + j] = matrix_[i][j];
      }
    }
    matrix_.resize(1);
    matrix_[0] = vector;
    column_ = column_ * row_;
    row_ = 1;
  }

  /**
   * @brief ベクトルのサイズを変形する
   */
  void Reshape(std::size_t row, std::size_t column)
  {
    assert(row * column == row_ * column_);
    row_ = row;
    column_ = column;

    for (std::size_t i = 0; i < row_; i++)
    {
      matrix_[i].resize(column_);
    }
  }


  /**
   * @brief 行列の転置を行う
   */
  self Transposed() const
  {
    self matrix(column_, row_);
    for (std::size_t i = 0; i < column_; i++)
    {
      for (std::size_t j = 0; j < row_; j++)
      {
        matrix[i][j] = matrix_[j][i];
      }
    }
    return matrix;
  }

  /**
   * @brief 2次元の行列を1次元のベクトル形式にする
   */
  self ToVector() const
  {
    row_array vector(column_ * row_);
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        vector[i * column_ + j] = matrix_[i][j];
      }
    }
    return self(vector, 1, row_ * column_);
  }

  /**
   * @brief 最大値を持つ要素のインデックスを取得する
   */
  std::pair<std::size_t, std::size_t> ArgMax() const
  {
    Primitive max = matrix_[0][0];
    std::size_t max_i = 0;
    std::size_t max_j = 0;
    for (std::size_t i = 0; i < row_; i++)
    {
      for (std::size_t j = 0; j < column_; j++)
      {
        if (max < matrix_[i][j])
        {
          max = matrix_[i][j];
          max_i = i;
          max_j = j;
        }
      }
    }
    return std::make_pair(max_i, max_j);
  }

  /**
 * @brief 全てランダムに初期化した行列を生成する
 */
  static MatrixN<Primitive> Randn(std::size_t row, std::size_t column)
  {
    MatrixN<Primitive> matrix(row, column);
    Random rand;
    for (std::size_t i = 0; i < row; i++)
    {
      for (std::size_t j = 0; j < column; j++)
      {
        matrix[i][j] = rand.uniform_real_distribution<Primitive>(0.F, 1.F);
      }
    }
    return matrix;
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
  for (std::size_t i = 0; i < matrix.Row(); i++)
  {
    if (i != 0)
    {
      os << " ";
    }
    os << "[";
    for (std::size_t j = 0; j < matrix.Column(); j++)
    {
      os << matrix[i][j];
      if (j != matrix.Column() - 1)
      {
        os << ", ";
      }
    }
    os << "]";
    if (i != matrix.Row() - 1)
    {
      os << std::endl;
    }
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

/**
 * @brief batch 処理に対応した行列の積を計算する
 */
template<typename Primitive>
MatrixN<Primitive> Dot(
  const std::vector<MatrixN<Primitive>> &lhv,
  const MatrixN<Primitive> &rhv)
{
  MatrixN<Primitive> matrix(lhv.size(), rhv.Column());
  for (std::size_t i = 0; i < lhv.size(); i++)
  {
    matrix[i] = Dot(lhv[i], rhv);
  }
  return matrix;
}

/**
 * @brief batch 処理に対応した行列の積を計算する
 */
template<typename Primitive>
MatrixN<Primitive> Dot(
  const MatrixN<Primitive> &lhv,
  const std::vector<MatrixN<Primitive>> &rhv)
{
  MatrixN<Primitive> matrix(lhv.Row(), rhv.size());
  for (std::size_t i = 0; i < rhv.size(); i++)
  {
    matrix[i] = Dot(lhv, rhv[i]);
  }
  return matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの加算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator+(
  const std::vector<MatrixN<Primitive>> &lhv,
  const MatrixN<Primitive> &rhv)
{
  MatrixN<Primitive> matrix(lhv.size(), rhv.Column());
  for (std::size_t i = 0; i < lhv.size(); i++)
  {
    matrix[i] = lhv[i] + rhv;
  }
  return matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの加算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator+(
  const MatrixN<Primitive> &lhv,
  const std::vector<MatrixN<Primitive>> &rhv)
{
  MatrixN<Primitive> matrix(lhv.Row(), rhv.size());
  for (std::size_t i = 0; i < rhv.size(); i++)
  {
    matrix[i] = lhv + rhv[i];
  }
  return matrix;
}

/**
 * @brief スカラーとの加算
 */
template<typename Primitive>
MatrixN<Primitive> operator+(
  Primitive lhv,
  const MatrixN<Primitive> &rhv
)
{
  MatrixN<Primitive> matrix(rhv.Row(), rhv.Column());
  for (std::size_t i = 0; i < rhv.Row(); i++)
  {
    for (std::size_t j = 0; j < rhv.Column(); j++)
    {
      matrix[i][j] = lhv + rhv[i][j];
    }
  }
  return matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの減算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator-(
  const std::vector<MatrixN<Primitive>>
  &lhv,
  const MatrixN<Primitive> &rhv
)
{
  MatrixN<Primitive> matrix(lhv.size(), rhv.Column());
  for (std::size_t i = 0; i < lhv.size(); i++)
  {
    matrix[i] = lhv[i] - rhv;
  }
  return matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの減算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator-(
  const MatrixN<Primitive> &lhv,
  const std::vector<MatrixN<Primitive>>
  &rhv)
{
  MatrixN<Primitive> matrix(lhv.Row(), rhv.size());
  for (
    std::size_t i = 0;
    i < rhv.
      size();
    i++)
  {
    matrix[i] = lhv - rhv[i];
  }
  return
    matrix;
}

template<typename Primitive>
MatrixN<Primitive> operator-(
  Primitive lhv,
  const MatrixN<Primitive> &rhv
)
{
  MatrixN<Primitive> matrix(rhv.Row(), rhv.Column());
  for (std::size_t i = 0; i < rhv.Row(); i++)
  {
    for (std::size_t j = 0; j < rhv.Column(); j++)
    {
      matrix[i][j] = lhv - rhv[i][j];
    }
  }
  return matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの乗算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator*(
  const std::vector<MatrixN<Primitive>>
  &lhv,
  const MatrixN<Primitive> &rhv
)
{
  MatrixN<Primitive> matrix(lhv.size(), rhv.Column());
  for (
    std::size_t i = 0;
    i < lhv.
      size();
    i++)
  {
    matrix[i] = lhv[i] *
      rhv;
  }
  return
    matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの乗算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator*(
  const MatrixN<Primitive> &lhv,
  const std::vector<MatrixN<Primitive>>
  &rhv)
{
  MatrixN<Primitive> matrix(lhv.Row(), rhv.size());
  for (
    std::size_t i = 0;
    i < rhv.
      size();
    i++)
  {
    matrix[i] =
      lhv * rhv[i];
  }
  return
    matrix;
}

template<typename Primitive>
MatrixN<Primitive> operator*(
  Primitive lhv,
  const MatrixN<Primitive> &rhv
)
{
  MatrixN<Primitive> matrix(rhv.Row(), rhv.Column());
  for (std::size_t i = 0; i < rhv.Row(); i++)
  {
    for (std::size_t j = 0; j < rhv.Column(); j++)
    {
      matrix[i][j] = lhv * rhv[i][j];
    }
  }
  return matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの除算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator/(
  const std::vector<MatrixN<Primitive>>
  &lhv,
  const MatrixN<Primitive> &rhv
)
{
  MatrixN<Primitive> matrix(lhv.size(), rhv.Column());
  for (
    std::size_t i = 0;
    i < lhv.
      size();
    i++)
  {
    matrix[i] = lhv[i] /
      rhv;
  }
  return
    matrix;
}

/**
 * @brief batch 処理に対応した行列の要素ごとの除算を計算する
 */
template<typename Primitive>
MatrixN<Primitive> operator/(
  const MatrixN<Primitive> &lhv,
  const std::vector<MatrixN<Primitive>>
  &rhv)
{
  MatrixN<Primitive> matrix(lhv.Row(), rhv.size());
  for (
    std::size_t i = 0;
    i < rhv.
      size();
    i++)
  {
    matrix[i] = lhv / rhv[i];
  }
  return
    matrix;
}

template<typename Primitive>
MatrixN<Primitive> operator/(
  Primitive lhv,
  const MatrixN<Primitive> &rhv
)
{
  MatrixN<Primitive> matrix(rhv.Row(), rhv.Column());
  for (std::size_t i = 0; i < rhv.Row(); i++)
  {
    for (std::size_t j = 0; j < rhv.Column(); j++)
    {
      matrix[i][j] = lhv / rhv[i][j];
    }
  }
  return matrix;
}


/**
 * @brief 行列の要素ごとに自然対数を計算する
 */
template<typename Primitive>
MatrixN<Primitive>
Log(const MatrixN<Primitive> &matrix)
{
  MatrixN<Primitive> result(matrix.Row(), matrix.Column());
  for (std::size_t i = 0; i < matrix.Row(); i++)
  {
    for (std::size_t j = 0; j < matrix.Column(); j++)
    {
      result[i][j] = std::log(matrix[i][j]);
    }
  }
  return result;
}

/**
 * @brief 行列の総和を計算する
 */
template<typename Primitive>
Primitive Sum(const MatrixN<Primitive> &matrix)
{
  Primitive sum = 0;
  for (std::size_t i = 0; i < matrix.Row(); i++)
  {
    for (std::size_t j = 0; j < matrix.Column(); j++)
    {
      sum += matrix[i][j];
    }
  }
  return sum;
}

/**
 * @brief 全てランダムに初期化した行列を生成する
 */
template<typename Primitive>
MatrixN<Primitive> Randn(std::size_t row, std::size_t column)
{
  MatrixN<Primitive> matrix(row, column);
  Random rand;
  for (std::size_t i = 0; i < row; i++)
  {
    for (std::size_t j = 0; j < column; j++)
    {
      matrix[i][j] = rand.normal_distribution<Primitive>(0.F, 1.F);
    }
  }
  return matrix;
}

template<typename Primitive>
std::pair<std::size_t, std::size_t> ArgMax(const MatrixN<Primitive> &matrix)
{
  Primitive max = matrix[0][0];
  std::size_t max_i = 0;
  std::size_t max_j = 0;
  for (std::size_t i = 0; i < matrix.Row(); i++)
  {
    for (std::size_t j = 0; j < matrix.Column(); j++)
    {
      if (max < matrix[i][j])
      {
        max = matrix[i][j];
        max_i = i;
        max_j = j;
      }
    }
  }
  return std::make_pair(max_i, max_j);
}

template<typename Primitive>
MatrixN<int> Equal(const MatrixN<Primitive> &a,
                   const MatrixN<Primitive> &b)
{
  MatrixN<int> matrix(a.Row(), a.Column());
  for (std::size_t i = 0; i < a.Row(); i++)
  {
    for (std::size_t j = 0; j < a.Column(); j++)
    {
      matrix[i][j] = (a[i][j] == b[i][j]) ? 1 : 0;
    }
  }
  return matrix;
}

} // namespace nagato

#endif //NAGATOLIB_SRC_MATRIX_N_HPP_

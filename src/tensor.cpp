//
// Created by toru on 2025/02/09.
//

#include "tensor.hpp"
#include <iostream>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <initializer_list>
#include <type_traits>

namespace nagato
{
Tensor::Tensor()
  : shape_(),
    strides_(),
    storage_()
{
}

Tensor::Tensor(const shape_type &shape)
  : shape_(shape),
    strides_(shape.size(), 1)
{
  if (!has_shape())
  {
    throw std::invalid_argument("shape is not set");
  }

  this->set_shape(shape);
}

bool Tensor::has_shape() const
{
  return !shape_.empty();
}

void Tensor::set_shape(const shape_type &shape)
{
  shape_ = shape;
  strides_.resize(shape.size(), 1);

  for (std::size_t i = shape.size() - 1; i > 0; --i)
  {
    strides_[i - 1] = strides_[i] * shape[i];
  }

  // 確保するストレージサイズを計算する
  std::size_t storage_size = 1;
  for (std::size_t i = 0; i < shape.size(); ++i)
  {
    storage_size *= shape[i];
  }

  // データ領域を確保
  storage_.resize(storage_size, 0);
}

const Tensor::shape_type &Tensor::shape() const
{
  return shape_;
}

const Tensor::strides_type &Tensor::strides() const
{
  return strides_;
}

const Tensor::storage_type &Tensor::storage() const
{
  return storage_;
}

Tensor::storage_type &Tensor::storage()
{
  return storage_;
}

Tensor Tensor::Fill(const shape_type &shape, const value_type &value)
{
  Tensor tensor(shape);
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < tensor.storage().size(); ++i) {
    tensor.storage()[i] = value;
  }
  return tensor;
}

Tensor Tensor::Zeros(const shape_type &shape)
{
  return Fill(shape, 0);
}

Tensor Tensor::Ones(const shape_type &shape)
{
  return Fill(shape, 1);
}

Tensor Tensor::Eye(const shape_type &shape)
{
  Tensor tensor(shape);
  std::fill(tensor.storage().begin(), tensor.storage().end(), 0);
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < shape[0]; ++i)
  {
    tensor(i, i) = 1;
  }
  return tensor;
}

void Tensor::IsSameShape(const Tensor &a, const Tensor &b)
{
  // ２つのテンソルの形状が等しいことをチェック
  if (a.shape() != b.shape())
  {
    for (std::size_t i = 0; i < a.shape().size(); ++i)
    {
      std::cerr << "a.shape()[i]: " << a.shape()[i] << ", b.shape()[i]: " << b.shape()[i] <<
        std::endl;
    }
    throw std::invalid_argument("tensor must have the same shape");
  }
  else
  {
    // すべての次元が等しいことをチェック
    for (std::size_t i = 0; i < a.shape().size(); ++i)
    {
      if (a.shape()[i] != b.shape()[i])
      {
        throw std::invalid_argument("tensor must have the same shape");
      }
    }
  }
}

Tensor operator+(const Tensor &a, const Tensor &b)
{
  return ApplyBroadcastBinaryOp(a, b, [](const float x, const float y) { return x + y; });
}

Tensor operator+(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i) {
    result.storage()[i] = a.storage()[i] + b;
  }
  return result;
}

Tensor operator+(const float &a, const Tensor &b)
{
  Tensor result(b.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < b.storage().size(); ++i) {
    result.storage()[i] = a + b.storage()[i];
  }
  return result;
}

Tensor operator-(const Tensor &a, const Tensor &b)
{
  return ApplyBroadcastBinaryOp(a, b, [](const float x, const float y) { return x - y; });
}

Tensor operator-(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i) {
    result.storage()[i] = a.storage()[i] - b;
  }
  return result;
}

Tensor operator-(const float &a, const Tensor &b)
{
  Tensor result(b.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < b.storage().size(); ++i) {
    result.storage()[i] = a - b.storage()[i];
  }
  return result;
}

Tensor operator*(const Tensor &a, const Tensor &b)
{
  return ApplyBroadcastBinaryOp(a, b, [](const float x, const float y) { return x * y; });
}

Tensor operator*(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i) {
    result.storage()[i] = a.storage()[i] * b;
  }
  return result;
}

Tensor operator*(const float &a, const Tensor &b)
{
  Tensor result(b.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < b.storage().size(); ++i) {
    result.storage()[i] = a * b.storage()[i];
  }
  return result;
}

Tensor operator/(const Tensor &a, const Tensor &b)
{
  return ApplyBroadcastBinaryOp(a, b, [](const float x, const float y) { return x / (y + 1e-7); });
}

Tensor operator/(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i) {
    result.storage()[i] = a.storage()[i] / (b + 1e-7);
  }
  return result;
}

Tensor operator/(const float &a, const Tensor &b)
{
  Tensor result(b.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < b.storage().size(); ++i) {
    result.storage()[i] = a / (b.storage()[i] + 1e-7);
  }
  return result;
}

Tensor Tensor::Reshape(const shape_type &new_shape)
{
  // 新しく生成される、その要素数が現在の要素すると等しいかチェック
  std::size_t new_size = std::accumulate(new_shape.begin(),
                                         new_shape.end(),
                                         1,
                                         std::multiplies<std::size_t>());

  if (new_size != storage_.size())
  {
    throw std::invalid_argument("new shape is not valid");
  }

  Tensor result(new_shape);
  std::copy(storage_.begin(), storage_.end(), result.storage().begin());
  return result;
}

Tensor Tensor::Dot(const Tensor &a, const Tensor &b)
{
  // 入力データの形状が等しいことをチェック
  if (a.shape() != b.shape())
  {
    for (std::size_t i = 0; i < a.shape().size(); ++i)
    {
      if (i < a.shape().size())
      {
        std::cerr << a.shape()[i] << ", ";
      }
      else
      {
        std::cerr << a.shape()[i] << std::endl;
      }

      if (i < b.shape().size())
      {
        std::cerr << b.shape()[i] << ", ";
      }
      else
      {
        std::cerr << b.shape()[i] << std::endl;
      }
    }
  }

  // ともに１次元のテンソルの場合
  if (a.shape().size() == 1)
  {
    Tensor result({1});
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      result(0) += a(i) * b(i);
    }
    return result;
  }

  // ともに２次元のテンソルの場合
  if (a.shape().size() == 2)
  {
    Tensor result({a.shape()[0]});
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        result(i) += a(i, j) * b(i, j);
      }
    }
    return result;
  }

  throw std::invalid_argument("input tensor must be a vector");
}

Tensor Tensor::Matmul(const Tensor &a, const Tensor &b)
{
  // 行列と行列の積
  if (a.shape().size() == 2 && b.shape().size() == 2)
  {
    // 入力データの形状をチェック
    if (a.shape()[1] != b.shape()[0])
    {
      for (const unsigned long i : a.shape())
      {
        std::cerr << i << ", ";
      }
      std::cerr << std::endl;
      for (const unsigned long i : b.shape())
      {
        std::cerr << i << ", ";
      }
      std::cerr << std::endl;
      throw std::invalid_argument("input tensor must have the same shape");
    }
    // 出力データの形状を計算
    shape_type result_shape = {a.shape()[0], b.shape()[1]};
    Tensor result(result_shape);

    // 行列積を計算する
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < b.shape()[1]; ++j)
      {
        for (std::size_t k = 0; k < b.shape()[0]; ++k)
        {
          result(i, j) += a(i, k) * b(k, j);
        }
      }
    }
    return result;
  }

  // バッチ行列とバッチ行列の積
  if (
    (a.shape().size() == 3) &&
    (b.shape().size() == 3)
  )
  {
    // 入力データの形状をチェック
    if (a.shape()[0] != b.shape()[0])
    {
      throw std::invalid_argument("batch size must be the same");
    }

    if (a.shape()[2] != b.shape()[1])
    {
      std::cerr << "a.shape()[0]: " << a.shape()[0] << ", a.shape()[1]: " << a.shape()[1] <<
        std::endl;
      std::cerr << "b.shape()[0]: " << b.shape()[0] << ", b.shape()[1]: " << b.shape()[1] <<
        std::endl;
      throw std::invalid_argument("input tensor must have the same shape");
    }

    // 出力データの形状を計算
    shape_type result_shape = {a.shape()[0], a.shape()[1], b.shape()[2]};
    Tensor result(result_shape);

    // 行列積を計算する
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < result_shape[0]; ++i) // バッチサイズ
    {
      for (std::size_t j = 0; j < result_shape[1]; ++j)
      {
        for (std::size_t k = 0; k < result_shape[2]; ++k)
        {
          for (std::size_t l = 0; l < b.shape()[1]; ++l)
          {
            result(i, j, k) += a(i, j, l) * b(i, l, k);
          }
        }
      }
    }
    return result;
  }

  throw std::invalid_argument("input tensor must be matrix");
}

Tensor Tensor::Sum(const Tensor &a)
{
  // 結果のテンソルを作成
  shape_type shape = a.shape();
  std::size_t shape_size = a.shape().size();
  shape.pop_back();
  if (shape_size == 1)
  {
    shape = {1};
  }
  Tensor result(shape);

  if (shape_size == 1)
  {
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      result(0) += a(i);
    }
  }
  else if (shape_size == 2)
  {
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        result(i) += a(i, j);
      }
    }
  }
  else if (shape_size == 3)
  {
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        for (std::size_t k = 0; k < a.shape()[2]; ++k)
        {
          result(i, j) += a(i, j, k);
        }
      }
    }
  }
  else
  {
    throw std::invalid_argument("tensor must be less than 4 dimensional");
  }

  return result;
}

Tensor Tensor::Sum(const Tensor &a, const std::size_t &axis)
{
  // 4次元以上のテンソルはサポートしない
  if (a.shape().size() >= 4)
  {
    throw std::invalid_argument("4次元以上のテンソルはSumでサポートされていません");
  }

  // 指定した軸が範囲内にあることを確認する
  if (axis >= a.shape().size())
  {
    throw std::invalid_argument("axis is out of range");
  }

  const auto &a_shape = a.shape();
  // 指定した軸を除いた新しいshapeを作成する
  std::vector<std::size_t> new_shape;
  for (std::size_t i = 0; i < a_shape.size(); ++i)
  {
    if (i == axis)
    {
      continue;
    }
    new_shape.push_back(a_shape[i]);
  }
  // new_shapeが空の場合はスカラーとみなし、shapeを{1}とする
  if (new_shape.empty())
  {
    new_shape.push_back(1);
  }

  // 結果のテンソルをゼロ初期化で作成する
  Tensor result = Tensor::Zeros(new_shape);

  // 軸に沿って総和を求めるため、前後のブロックサイズを計算する
  std::size_t pre = 1;
  for (std::size_t i = 0; i < axis; ++i)
  {
    pre *= a_shape[i];
  }
  std::size_t d = a_shape[axis];
  std::size_t post = 1;
  for (std::size_t i = axis + 1; i < a_shape.size(); ++i)
  {
    post *= a_shape[i];
  }

  // テンソル内部はrow-majorと仮定し、
  // 元のテンソルを(pre, d, post)とみなしてd方向に沿って総和を計算する
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < pre; ++i)
  {
    for (std::size_t j = 0; j < post; ++j)
    {
      float sum = 0;
      for (std::size_t k = 0; k < d; ++k)
      {
        std::size_t index = i * (d * post) + k * post + j;
        sum += a.storage()[index];
      }
      std::size_t result_index = i * post + j;
      result.storage()[result_index] = sum;
    }
  }

  return result;
}

Tensor Tensor::Sigmoid(const Tensor &a)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < result.storage().size(); ++i)
  {
    result.storage()[i] = 1.f / (1 + std::exp(-a.storage()[i]) + 1e-7);
  }
  return result;
}

Tensor Tensor::ReLU(const Tensor &a)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.shape()[0]; ++i)
  {
    result(i) = std::max(0.0f, a(i));
  }
  return result;
}

Tensor Tensor::Exp(const Tensor &a)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i) {
    result.storage()[i] = std::exp(a.storage()[i]);
  }
  return result;
}

Tensor Tensor::Log(const Tensor &a)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i) {
    result.storage()[i] = std::log(a.storage()[i] + 1e-7);
  }
  return result;
}

Tensor Tensor::Softmax(const Tensor &a)
{
  const std::size_t shape_size = a.shape().size();
  if (shape_size > 2) {
    throw std::invalid_argument("tensor must be less than 3 dimensional");
  }

  if (shape_size == 1) {
    // 1次元の場合、入力全体の最大値を引いてからexpを計算
    float max_val = *std::max_element(a.storage().begin(), a.storage().end());
    Tensor shifted(a.shape());
    for (std::size_t i = 0; i < a.storage().size(); ++i) {
      shifted.storage()[i] = a.storage()[i] - max_val;
    }
    Tensor exp_a = Exp(shifted);
    float sum = std::accumulate(exp_a.storage().begin(), exp_a.storage().end(), 0.0f);
    Tensor result(a.shape());
    for (std::size_t i = 0; i < result.storage().size(); ++i) {
      result.storage()[i] = exp_a.storage()[i] / sum;
    }
    return result;
  }

  if (shape_size == 2) {
    const auto &a_shape = a.shape();
    Tensor result(a_shape);
    // 行単位に計算する
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a_shape[0]; ++i) {
      // 各行の最大値を計算
      float row_max = a(i, 0);
      for (std::size_t j = 1; j < a_shape[1]; ++j) {
        if (a(i, j) > row_max) {
          row_max = a(i, j);
        }
      }
      // 最大値を引いて exp を計算
      float row_sum = 0;
      for (std::size_t j = 0; j < a_shape[1]; ++j) {
        float shifted = a(i, j) - row_max;
        float exp_val = std::exp(shifted);
        result(i, j) = exp_val; // 一旦 exp の値を格納
        row_sum += exp_val;
      }
      // 正規化
      for (std::size_t j = 0; j < a_shape[1]; ++j) {
        result(i, j) /= row_sum;
      }
    }
    return result;
  }

  throw std::invalid_argument("tensor must be less than 3 dimensional");
}

void Tensor::Print(const Tensor &a)
{
  if (a.shape().size() == 1)
  {
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      std::cout << a(i) << ", ";
    }
    std::cout << std::endl;
  }
  if (a.shape().size() == 2)
  {
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        std::cout << a(i, j) << ", ";
      }
      std::cout << std::endl;
    }
  }
  if (a.shape().size() == 3)
  {
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        for (std::size_t k = 0; k < a.shape()[2]; ++k)
        {
          std::cout << a(i, j, k) << ", ";
        }
      }
    }
  }

  if (a.shape().size() >= 4)
  {
    throw std::invalid_argument("tensor must be less than 4 dimensional");
  }

  std::cout << std::endl;
}

void Tensor::PrintShape(const Tensor &a)
{
  for (unsigned long i : a.shape())
  {
    std::cout << i << ", ";
  }
  std::cout << std::endl;
}

Tensor Tensor::Random(const shape_type &shape)
{
  Tensor result(shape);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<value_type> dis(-1.0, 1.0);
  for (std::size_t i = 0; i < result.storage().size(); ++i)
  {
    result.storage()[i] = dis(gen);
  }
  return result;
}

Tensor Tensor::RandomNormal(const shape_type &shape)
{
  Tensor result(shape);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<value_type> dis(0.0, 1.0);
  std::transform(
    result.storage().begin(),
    result.storage().end(),
    result.storage().begin(),
    [&dis, &gen](const value_type &x) { return dis(gen); }
  );
  return result;
}

Tensor Tensor::operator-() const
{
  Tensor result(shape_);
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < storage_.size(); ++i) {
    result.storage()[i] = -storage_[i];
  }
  return result;
}

Tensor Tensor::FromArray(const std::vector<value_type> &array)
{
  Tensor result({array.size()});
  std::copy(array.begin(), array.end(), result.storage().begin());
  return result;
}

Tensor Tensor::FromArray(const std::vector<std::vector<value_type> > &array)
{
  Tensor result({array.size(), array[0].size()});

  // すべての行が同じサイズであることをチェック
  for (const auto &row : array)
  {
    if (row.size() != array[0].size())
    {
      throw std::invalid_argument("all rows must have the same size");
    }
  }

  for (std::size_t i = 0; i < array.size(); ++i)
  {
    std::copy(array[i].begin(), array[i].end(), result.storage().begin() + i * array[i].size());
  }
  return result;
}

Tensor Tensor::FromArray(const std::vector<std::vector<std::vector<value_type> > > &array)
{
  // 3次元テンソルとして初期化
  Tensor result({array.size(), array[0].size(), array[0][0].size()});

  // すべての行が同じサイズであることをチェック
  for (const auto &row : array)
  {
    if (row.size() != array[0].size())
    {
      throw std::invalid_argument("all rows must have the same size");
    }
    // すべての列が同じサイズであることをチェック
    for (const auto &col : row)
    {
      if (col.size() != array[0][0].size())
      {
        throw std::invalid_argument("all columns must have the same size");
      }
    }
  }

  // 各次元のサイズ
  std::size_t d1 = array[0].size();
  std::size_t d2 = array[0][0].size();

#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < array.size(); ++i)
  {
    for (std::size_t j = 0; j < d1; ++j)
    {
      // 正しいオフセット: 第1次元 i に対して d1*d2、さらに第2次元 j に対して d2
      std::copy(
        array[i][j].begin(),
        array[i][j].end(),
        result.storage().begin() + i * (d1 * d2) + j * d2);
    }
  }
  return result;
}

Tensor Tensor::Transform(const Tensor &a, const std::function<value_type(value_type)> &func)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i) {
    result.storage()[i] = func(a.storage()[i]);
  }
  return result;
}

Tensor Tensor::Transpose(const Tensor &a)
{
  // 2, 3 以外の軸数に時は転置しない
  if (a.shape().size() != 2 && a.shape().size() != 3)
  {
    throw std::invalid_argument("tensor must be 2 or 3 dimensional");
  }

  if (a.shape().size() == 2)
  {
    // 転置後の形状を計算
    shape_type shape = {a.shape()[1], a.shape()[0]};

    // 転置後のテンソルを作成
    Tensor result(shape);

#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        result(j, i) = a(i, j);
      }
    }

    return result;
  }
  else if (a.shape().size() == 3)
  {
    // 転置後の形状を計算
    shape_type shape = {a.shape()[0], a.shape()[2], a.shape()[1]};

    // 転置後のテンソルを作成
    Tensor result(shape);

#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        for (std::size_t k = 0; k < a.shape()[2]; ++k)
        {
          result(i, k, j) = a(i, j, k);
        }
      }
    }

    return result;
  }

  throw std::invalid_argument("tensor must be 2 or 3 dimensional");
}

Tensor Tensor::Slice(const std::size_t &axis) const
{
  // 指定した軸が範囲内にあることを確認する
  if (axis < 0 || shape_[0] <= axis)
  {
    throw std::invalid_argument("axis is out of range");
  }

  // 分割したテンソルのshape を計算する
  shape_type shape(shape_.begin() + 1, shape_.end());

  // 分割したテンソルを作成する
  Tensor result(shape);

  // 分割したテンソルを作成する
  const std::size_t start_index = axis * strides_[0];
  const std::size_t end_index = start_index + strides_[0];

  std::copy(
    storage_.begin() + start_index,
    storage_.begin() + end_index,
    result.storage().begin()
  );

  return result;
}

Tensor Tensor::Slice(const std::size_t &start, const std::size_t &end) const
{
  // 指定した範囲が範囲内にあることを確認する
  if (start < 0 || end > shape_[0])
  {
    throw std::invalid_argument("start and end must be within the range of the tensor");
  }

  // end が start より大きいことを確認する
  if (end <= start)
  {
    throw std::invalid_argument("end must be greater than start");
  }

  // 分割したテンソルのshape を計算する
  shape_type shape(this->shape_);
  const std::size_t size = end - start + 1;
  shape[0] = size;

  // 分割したテンソルを作成する
  Tensor result(shape);

  // 分割したテンソルを作成する
  const std::size_t start_index = start * strides_[0];
  const std::size_t end_index = start_index + size * strides_[0];
  std::copy(
    storage_.begin() + start_index,
    storage_.begin() + end_index,
    result.storage().begin());

  return result;
}

Tensor Tensor::Argmax() const
{
  // 1次元のテンソルの場合
  if (shape_.size() == 1)
  {
    Tensor result({1});
    result(0) = std::distance(
      storage_.begin(),
      std::max_element(storage_.begin(), storage_.end()));
    return result;
  }

  // 2次元のテンソルの場合
  if (shape_.size() == 2)
  {
    Tensor result({shape_[0]});
    for (std::size_t i = 0; i < shape_[0]; ++i)
    {
      Tensor slice = this->Slice(i);
      result(i) = slice.Argmax()(0);
    }
    return result;
  }

  throw std::invalid_argument("tensor must be 1 or 2 dimensional");
}

bool Tensor::Equal(const Tensor &a, const Tensor &b)
{
  if (a.shape() != b.shape())
  {
    return false;
  }

  for (std::size_t i = 0; i < a.storage().size(); ++i)
  {
    if (a.storage()[i] != b.storage()[i])
    {
      return false;
    }
  }
  return true;
}

Tensor operator==(const Tensor &a, const Tensor &b)
{
  Tensor result(a.shape());
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t i = 0; i < a.storage().size(); ++i)
  {
    result.storage()[i] = a.storage()[i] == b.storage()[i] ? 1.0 : 0.0;
  }
  return result;
}

Tensor Tensor::FromCSV(const std::string &filename)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    throw std::runtime_error("ファイルを開くことができませんでした: " + filename);
  }
  std::string line;
  std::vector<std::vector<float> > data;
  while (std::getline(file, line))
  {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
    {
      row.push_back(std::stof(cell));
    }
    data.push_back(row);
  }
  return FromArray(data);
}

Tensor Tensor::Mean(const Tensor &a)
{
  // ベクトルの場合
  if (a.shape().size() == 1)
  {
    Tensor result({1});
    // 分母が0の場合は例外を送出
    if (a.shape()[0] == 0)
    {
      throw std::invalid_argument("denominator is 0");
    }
    result(0) = std::accumulate(a.storage().begin(), a.storage().end(), 0.0f) / a.shape()[0];
    return result;
  }

  // 行列の場合
  if (a.shape().size() == 2)
  {
    Tensor result({a.shape()[0]});
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      // 行をベクトルとして取り出して計算する
      Tensor slice = a.Slice(i);
      result(i) = Tensor::Mean(slice)(0);
    }
    return result;
  }

  throw std::invalid_argument("tensor must be 1 or 2 dimensional");
}

Tensor Tensor::Abs(const Tensor &a)
{
  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 result.storage().begin(),
                 [](float x) { return std::abs(x); });
  return result;
}

int Tensor::IsBroadcastable(const Tensor &a, const Tensor &b)
{
  // それぞれのテンソルの shape（次元数を格納した vector）を取得
  const auto &shapeA = a.shape();
  const auto &shapeB = b.shape();
  std::size_t rankA = shapeA.size();
  std::size_t rankB = shapeB.size();

  // ブロードキャストは末尾の次元から判定するので、最大ランクに合わせる
  std::size_t maxRank = std::max(rankA, rankB);

  // 末尾から各次元のサイズを取り出して比較する
  for (std::size_t i = 0; i < maxRank; ++i)
  {
    // i=0 では最後の次元、i=1 ではそのひとつ前の次元…となる
    std::size_t dimA = (i < rankA) ? shapeA[rankA - 1 - i] : 1;
    std::size_t dimB = (i < rankB) ? shapeB[rankB - 1 - i] : 1;

    // どちらの次元も同じか、またはどちらかが1であればブロードキャスト可能
    if (dimA != dimB && dimA != 1 && dimB != 1)
    {
      return 0;
    }
  }
  return true;
}

template<typename BinaryOp>
Tensor ApplyBroadcastBinaryOp(const Tensor &a, const Tensor &b, BinaryOp op)
{
  // まず、2つのテンソルがブロードキャスト可能かチェックする
  if (!Tensor::IsBroadcastable(a, b))
  {
    throw std::invalid_argument("Tensors are not broadcastable");
  }

  // もし形状が完全に一致していれば、単純な要素ごとのループで処理する
  if (a.shape() == b.shape())
  {
    Tensor result(a.shape());
    const auto &a_storage = a.storage();
    const auto &b_storage = b.storage();
    auto &result_storage = result.storage();
    for (std::size_t i = 0; i < a_storage.size(); ++i)
    {
      result_storage[i] = op(a_storage[i], b_storage[i]);
    }
    return result;
  }

  // 形状が一致しない場合、ブロードキャスト処理を行う

  // 元のshapeを取得
  const auto &shapeA = a.shape();
  const auto &shapeB = b.shape();
  std::size_t rankA = shapeA.size();
  std::size_t rankB = shapeB.size();
  std::size_t maxRank = std::max(rankA, rankB);

  // 先頭側に1を補完してパディングしたshapeを作成
  std::vector<std::size_t> paddedA(maxRank, 1);
  std::vector<std::size_t> paddedB(maxRank, 1);
  for (std::size_t i = 0; i < rankA; ++i)
  {
    paddedA[maxRank - rankA + i] = shapeA[i];
  }
  for (std::size_t i = 0; i < rankB; ++i)
  {
    paddedB[maxRank - rankB + i] = shapeB[i];
  }

  // 各軸ごとに拡張後のサイズは、paddedA と paddedB の大きい方となる
  std::vector<std::size_t> broadcastShape(maxRank);
  for (std::size_t i = 0; i < maxRank; ++i)
  {
    broadcastShape[i] = std::max(paddedA[i], paddedB[i]);
  }
  Tensor result(broadcastShape);

  // 拡張後の総要素数を計算
  std::size_t total = 1;
  for (auto dim : broadcastShape)
  {
    total *= dim;
  }

  // 補完後のshapeからストライドを計算するラムダ（row-major順）
  auto computeStrides = [](const std::vector<std::size_t> &shape) -> std::vector<std::size_t>
  {
    std::vector<std::size_t> strides(shape.size());
    if (!shape.empty())
    {
      strides[shape.size() - 1] = 1;
      for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
      {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
    }
    return strides;
  };

  std::vector<std::size_t> stridesA = computeStrides(paddedA);
  std::vector<std::size_t> stridesB = computeStrides(paddedB);

  // flat index を multi-index に変換するラムダ
  auto flatToMultiIndex = [&](std::size_t flatIndex,
                              const std::vector<std::size_t> &shape) -> std::vector<std::size_t>
  {
    std::vector<std::size_t> indices(shape.size(), 0);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
      indices[i] = flatIndex % shape[i];
      flatIndex /= shape[i];
    }
    return indices;
  };

  // 各要素に対して、ブロードキャストに従ったインデックス変換を行い、opを適用する
#ifdef NAGATO_OPENMP
  #pragma omp parallel for
#endif
  for (std::size_t idx = 0; idx < total; ++idx)
  {
    std::vector<std::size_t> multiIndex = flatToMultiIndex(idx, broadcastShape);
    std::size_t indexA = 0;
    std::size_t indexB = 0;
    for (std::size_t i = 0; i < maxRank; ++i)
    {
      // 対象の次元が1の場合、常に0番目の要素を参照する
      std::size_t idxA = (paddedA[i] == 1 ? 0 : multiIndex[i]);
      std::size_t idxB = (paddedB[i] == 1 ? 0 : multiIndex[i]);
      indexA += idxA * stridesA[i];
      indexB += idxB * stridesB[i];
    }
    result.storage()[idx] = op(a.storage()[indexA], b.storage()[indexB]);
  }
  return result;
}

Tensor Tensor::Concat(const std::vector<Tensor> &tensors)
{
  if (tensors.empty())
  {
    throw std::invalid_argument("Tensor vector is empty");
  }

  // すべてのテンソルの形状が同一であることをチェック
  const auto &firstShape = tensors[0].shape();
  for (const auto &tensor : tensors)
  {
    if (tensor.shape() != firstShape)
    {
      throw std::invalid_argument("all tensors must have the same shape");
    }
  }

  // 新しいテンソルの形状を構築する
  // 先頭にテンソル数、続いて各テンソルの元の形状を追加する
  shape_type new_shape = {tensors.size()};
  new_shape.insert(new_shape.end(), firstShape.begin(), firstShape.end());

  // 結果テンソルを作成 (new_shapeに基づいてストレージサイズも自動的に確保される)
  Tensor result(new_shape);

  // 各テンソルのストレージを連結してコピーする
  std::size_t offset = 0;
#ifdef NAGATO_OPENMP
    #pragma omp parallel for
#endif
  for (const auto &tensor : tensors)
  {
    const auto &storage = tensor.storage();
    std::copy(storage.begin(), storage.end(), result.storage().begin() + offset);
    offset += storage.size();
  }
  return result;
}

Tensor::value_type Tensor::Max(const Tensor &a)
{
  return *std::max_element(a.storage().begin(), a.storage().end());
}

Tensor::value_type Tensor::Min(const Tensor &a)
{
  return *std::min_element(a.storage().begin(), a.storage().end());
}

bool Tensor::IsNan(const Tensor &a)
{
  return std::any_of(a.storage().begin(), a.storage().end(), [](const value_type &x) { return std::isnan(x); });
}




} // namespace nagato

//
// Created by toru on 2025/02/09.
//

#include "tensor.hpp"
#include <iostream>
#include <numeric>
#include <random>

namespace nagato
{
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
  std::fill(tensor.storage().begin(), tensor.storage().end(), value);
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
  Tensor::IsSameShape(a, b);
  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 b.storage().begin(),
                 result.storage().begin(),
                 std::plus<float>());
  return result;
}

Tensor operator+(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 result.storage().begin(),
                 [b](const float x) { return x + b; });
  return result;
}

Tensor operator-(const Tensor &a, const Tensor &b)
{
  Tensor::IsSameShape(a, b);

  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 b.storage().begin(),
                 result.storage().begin(),
                 std::minus<float>());
  return result;
}

Tensor operator-(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 result.storage().begin(),
                 [b](const float x) { return x - b; });
  return result;
}

Tensor operator*(const Tensor &a, const Tensor &b)
{
  Tensor::IsSameShape(a, b);

  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 b.storage().begin(),
                 result.storage().begin(),
                 std::multiplies<float>());
  return result;
}

Tensor operator*(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 result.storage().begin(),
                 [b](const float x) { return x * b; });
  return result;
}

Tensor operator/(const Tensor &a, const Tensor &b)
{
  Tensor::IsSameShape(a, b);

  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 b.storage().begin(),
                 result.storage().begin(),
                 std::divides<float>());
  return result;
}

Tensor operator/(const Tensor &a, const float &b)
{
  Tensor result(a.shape());
  std::transform(a.storage().begin(),
                 a.storage().end(),
                 result.storage().begin(),
                 [b](const float x) { return x / b; });
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
    throw std::invalid_argument("input tensor must have the same shape");
  }

  // ともに１次元のテンソルの場合
  if (a.shape().size() == 1)
  {
    Tensor result({1});
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
      throw std::invalid_argument("input tensor must have the same shape");
    }

    // 出力データの形状を計算 
    shape_type result_shape = {a.shape()[0], b.shape()[1]};
    Tensor result(result_shape);

    // 行列積を計算する
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
      throw std::invalid_argument("input tensor must have the same shape");
    }

    // 出力データの形状を計算
    shape_type result_shape = {a.shape()[0], a.shape()[1], b.shape()[2]};
    Tensor result(result_shape);

    // 行列積を計算する
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
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      result(0) += a(i);
    }
  }
  else if (shape_size == 2)
  {
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

Tensor Tensor::Sigmoid(const Tensor &a)
{
  Tensor result(a.shape());
  for (std::size_t i = 0; i < result.storage().size(); ++i)
  {
    result.storage()[i] = 1.f / (1 + std::exp(-a.storage()[i]) + 1e-7);
  }
  return result;
}

Tensor Tensor::ReLU(const Tensor &a)
{
  Tensor result(a.shape());
  for (std::size_t i = 0; i < a.shape()[0]; ++i)
  {
    result(i) = std::max(0.0f, a(i));
  }
  return result;
}

Tensor Tensor::Exp(const Tensor &a)
{
  Tensor result(a.shape());
  std::transform(
    a.storage().begin(),
    a.storage().end(),
    result.storage().begin(),
    [](const float x) { return std::exp(x); }
  );
  return result;
}

Tensor Tensor::Log(const Tensor &a)
{
  Tensor result(a.shape());
  std::transform(
    a.storage().begin(),
    a.storage().end(),
    result.storage().begin(),
    [](const Tensor::value_type &x) { return std::log(x); }
  );
  return result;
}

Tensor Tensor::Softmax(const Tensor &a)
{
  const std::size_t shape_size = a.shape().size();

  if (shape_size > 2)
  {
    throw std::invalid_argument("tensor must be less than 3 dimensional");
  }

  Tensor exp_a = Exp(a);

  if (shape_size == 1)
  {
    Tensor result(a.shape());
    float sum = 0;
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      sum += exp_a(i);
    }

    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      result(i) = exp_a(i) / sum;
    }
    return result;
  }

  if (shape_size == 2)
  {
    // 総和を求める
    Tensor sum = Tensor::Sum(exp_a);
    Tensor result(a.shape());
    for (std::size_t i = 0; i < a.shape()[0]; ++i)
    {
      for (std::size_t j = 0; j < a.shape()[1]; ++j)
      {
        result(i, j) = exp_a(i, j) / sum(i);
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

Tensor Tensor::operator-() const
{
  Tensor result(shape_);
  std::transform(
    storage_.begin(), 
    storage_.end(), 
    result.storage().begin(), 
    [](const value_type &x) { return -x; }
    );
  return result;
}

} // namespace nagato

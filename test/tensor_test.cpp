#include <gtest/gtest.h>

#include "nagatolib.hpp"
using namespace nagato;

TEST(TensorTest, Constructor)
{
  Tensor tensor({2, 3});
  EXPECT_EQ(tensor.shape(), (std::vector<std::size_t>{2, 3}));
}

// 要素アクセス
TEST(TensorTest, operator)
{
  Tensor tensor({2, 2});

  // shape のサイズが 2 であるかチェック
  EXPECT_EQ(tensor.shape().size(), 2);

  // storage のサイズが 4 であるかチェック
  EXPECT_EQ(tensor.storage().size(), 4);

  tensor(0, 0) = 1;
  tensor(0, 1) = 2;
  tensor(1, 0) = 3;
  tensor(1, 1) = 4;

  const std::vector<float> expected = {1, 2, 3, 4};
  const std::vector<float> actual = tensor.storage();
  EXPECT_EQ(actual, expected);
}

// 要素アクセステスト (3次元)
TEST(TensorTest, operator3D)
{
  Tensor tensor({2, 2, 2});

  // shape のサイズが 3 であるかチェック
  EXPECT_EQ(tensor.shape().size(), 3);

  // storage のサイズが 8 であるかチェック
  EXPECT_EQ(tensor.storage().size(), 8);

  for (std::size_t i = 0; i < 2; ++i)
  {
    for (std::size_t j = 0; j < 2; ++j)
    {
      for (std::size_t k = 0; k < 2; ++k)
      {
        tensor(i, j, k) = i * 4 + j * 2 + k;
      }
    }
  }

  const std::vector<float> expected = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<float> actual = tensor.storage();
  EXPECT_EQ(actual, expected);
}

// Fill のテスト
TEST(TensorTest, fill)
{
  Tensor tensor = Tensor::Fill({2, 2}, 1);
  EXPECT_EQ(tensor.storage(), (std::vector<float>{1, 1, 1, 1}));
}

// Fill のテスト (3次元)
TEST(TensorTest, fill3D)
{
  Tensor tensor = Tensor::Fill({2, 2, 2}, 1);
  EXPECT_EQ(tensor.storage(), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1}));
}

// zeros テスト
TEST(TensorTest, zeros)
{
  Tensor tensor = Tensor::Zeros({2, 2});
  EXPECT_EQ(tensor.storage().size(), 4);
  EXPECT_EQ(tensor.storage(), (std::vector<float>{0, 0, 0, 0}));
}

// ones テスト
TEST(TensorTest, ones)
{
  Tensor tensor = Tensor::Ones({2, 2});
  EXPECT_EQ(tensor.storage().size(), 4);
  EXPECT_EQ(tensor.storage(), (std::vector<float>{1, 1, 1, 1}));
}

// ones テスト (3次元)
TEST(TensorTest, ones3D)
{
  Tensor tensor = Tensor::Ones({2, 2, 2});
  EXPECT_EQ(tensor.storage().size(), 8);
  EXPECT_EQ(tensor.storage(), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1}));
}

// eye テスト
TEST(TensorTest, eye)
{
  Tensor tensor = Tensor::Eye({2, 2});
  EXPECT_EQ(tensor.storage().size(), 4);
  EXPECT_EQ(tensor.storage(), (std::vector<float>{1, 0, 0, 1}));
}

// 加算テスト
TEST(TensorTest, add)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a + b;
  EXPECT_EQ(c.storage(), (std::vector<float>{2, 2, 2, 2}));
}

// 減算テスト
TEST(TensorTest, sub)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a - b;
  EXPECT_EQ(c.storage(), (std::vector<float>{0, 0, 0, 0}));
}

// 乗算テスト
TEST(TensorTest, mul)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a * b;
  EXPECT_EQ(c.storage(), (std::vector<float>{1, 1, 1, 1}));
}

// 除算テスト
TEST(TensorTest, div)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a / b;
  EXPECT_EQ(c.storage(), (std::vector<float>{1, 1, 1, 1}));
}

// reshape テスト
TEST(TensorTest, reshape)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = a.Reshape({4});
  EXPECT_EQ(b.storage(), (std::vector<float>{1, 1, 1, 1}));

  // 新しい形状が期待したものと同じかチェック
  EXPECT_EQ(b.shape(), (std::vector<std::size_t>{4}));
}

// reshape テスト (3次元)
TEST(TensorTest, reshape3D)
{
  Tensor a = Tensor::Ones({2, 2, 2});
  Tensor b = a.Reshape({4, 2});
  EXPECT_EQ(b.storage(), (std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1}));

  // 新しい形状が期待したものと同じかチェック
  EXPECT_EQ(b.shape(), (std::vector<std::size_t>{4, 2}));
}

// dot テスト
// 1次元のテンソル同士
TEST(TensorTest, dot1D)
{
  Tensor a = Tensor::Ones({2});
  Tensor b = Tensor::Ones({2});
  Tensor c = Tensor::Dot(a, b);
  EXPECT_EQ(c.storage(), (std::vector<float>{2}));
}

// 2次元のテンソル同士
TEST(TensorTest, dot2D)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = Tensor::Dot(a, b);
  EXPECT_EQ(c.storage(), (std::vector<float>{2, 2}));
}

// 行列積 (Matmul) テスト
TEST(TensorTest, matmul)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = Tensor::Matmul(a, b);
  Tensor ans = Tensor::Zeros({2, 2});

  ans(0, 0) = 2;
  ans(0, 1) = 2;
  ans(1, 0) = 2;
  ans(1, 1) = 2;

  EXPECT_EQ(c.storage(), ans.storage());
}

// 行列積 (Matmul) テスト (3次元)
TEST(TensorTest, matmul3D)
{
  Tensor a = Tensor::Ones({2, 2, 2});
  Tensor b = Tensor::Ones({2, 2, 2});

  // shape をチェック
  EXPECT_EQ(a.shape().size(), 3);
  EXPECT_EQ(b.shape().size(), 3);

  Tensor c = Tensor::Matmul(a, b);
  Tensor ans = Tensor::Zeros({2, 2, 2});

  ans(0, 0, 0) = 2;
  ans(0, 0, 1) = 2;
  ans(0, 1, 0) = 2;
  ans(0, 1, 1) = 2;
  ans(1, 0, 0) = 2;
  ans(1, 0, 1) = 2;
  ans(1, 1, 0) = 2;
  ans(1, 1, 1) = 2;

  EXPECT_EQ(c.storage(), ans.storage());
}

// Matmul のテスト (3x2 * 2x3)
TEST(TensorTest, matmul3x2x2x3)
{
  Tensor a = Tensor::Ones({3, 2});
  for (std::size_t i = 0; i < 3; ++i)
  {
    for (std::size_t j = 0; j < 2; ++j)
    {
      a(i, j) = i * 2 + j + 1;
    }
  }

  Tensor b = Tensor::Ones({2, 3});
  for (std::size_t i = 0; i < 2; ++i)
  {
    for (std::size_t j = 0; j < 3; ++j)
    {
      b(i, j) = i * 3 + j + 1;
    }
  }

  Tensor c = Tensor::Matmul(a, b);
  Tensor ans = Tensor::Zeros({3, 3});
  ans(0, 0) = 9;
  ans(0, 1) = 12;
  ans(0, 2) = 15;
  ans(1, 0) = 19;
  ans(1, 1) = 26;
  ans(1, 2) = 33;
  ans(2, 0) = 29;
  ans(2, 1) = 40;
  ans(2, 2) = 51;

  EXPECT_EQ(c.storage(), ans.storage());
}

// Matmul のテスト (2x3x2 * 2x2x3)
TEST(TensorTest, matmul2x3x2x2x3)
{
  Tensor a = Tensor::Ones({2, 3, 2});
  for (std::size_t i = 0; i < 2; ++i)
  {
    for (std::size_t j = 0; j < 3; ++j)
    {
      for (std::size_t k = 0; k < 2; ++k)
      {
        a(i, j, k) = j * 2 + k + 1;
      }
    }
  }

  Tensor b = Tensor::Ones({2, 2, 3});
  for (std::size_t i = 0; i < 2; ++i)
  {
    for (std::size_t j = 0; j < 2; ++j)
    {
      for (std::size_t k = 0; k < 3; ++k)
      {
        b(i, j, k) = j * 3 + k + 1;
      }
    }
  }

  Tensor c = Tensor::Matmul(a, b);
  Tensor ans = Tensor::Zeros({2, 3, 3});
  ans(0, 0, 0) = 9;
  ans(0, 0, 1) = 12;
  ans(0, 0, 2) = 15;
  ans(0, 1, 0) = 19;
  ans(0, 1, 1) = 26;
  ans(0, 1, 2) = 33;
  ans(0, 2, 0) = 29;
  ans(0, 2, 1) = 40;
  ans(0, 2, 2) = 51;
  ans(1, 0, 0) = 9;
  ans(1, 0, 1) = 12;
  ans(1, 0, 2) = 15;
  ans(1, 1, 0) = 19;
  ans(1, 1, 1) = 26;
  ans(1, 1, 2) = 33;
  ans(1, 2, 0) = 29;
  ans(1, 2, 1) = 40;
  ans(1, 2, 2) = 51;

  EXPECT_EQ(c.storage(), ans.storage());
}

// sum テスト (1次元)
TEST(TensorTest, sum1D)
{
  Tensor a = Tensor::Ones({2});
  Tensor c = Tensor::Sum(a);
  EXPECT_EQ(c.storage(), (std::vector<float>{2}));
}

// sum テスト (2次元)
TEST(TensorTest, sum2D)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor c = Tensor::Sum(a);
  EXPECT_EQ(c.storage(), (std::vector<float>{2, 2}));
}

// sum テスト (3次元)
TEST(TensorTest, sum3D)
{
  Tensor a = Tensor::Ones({2, 2, 2});
  Tensor c = Tensor::Sum(a);
  EXPECT_EQ(c.storage(), (std::vector<float>{2, 2, 2, 2}));
}

#include <gtest/gtest.h>

#include "nagatolib.hpp"
using namespace nagato;

/**
 * @brief 重みパラメータに対する勾配を計算する関数. 
 * @note バッチ処理に対応している
 * @param func 勾配を計算する関数
 * @param x 入力. 一番最初の次元はバッチサイズ
 * @return 勾配
 */
Tensor numerical_gradient(std::function<Tensor(const Tensor &)> func, const Tensor &x)
{
  constexpr Tensor::value_type h = 1e-3;

  // x と同じ形状を持つゼロ初期化のテンソルを作成する
  Tensor grad = Tensor::Zeros(x.shape());
  // x の変更可能なコピーを作成する
  Tensor x_copy = x;

  // Tensor のストレージ全体（全要素）でループ
  for (std::size_t idx = 0; idx < x_copy.storage().size(); ++idx)
  {
    // 現在の値を記憶
    Tensor::value_type tmp_val = x_copy.storage()[idx];

    // x + h における f の値を計算
    x_copy.storage()[idx] = tmp_val + h;
    Tensor fxh1 = func(x_copy);

    // x - h における f の値を計算
    x_copy.storage()[idx] = tmp_val - h;
    Tensor fxh2 = func(x_copy);

    // 値を元に戻す
    x_copy.storage()[idx] = tmp_val;

    // 中心差分による数値勾配を計算
    // ※ここでは、func がスカラー値 (1要素のTensor) を返すと仮定しています。
    grad.storage()[idx] = (fxh1.storage()[0] - fxh2.storage()[0]) / (2 * h);
  }

  return grad;
}

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

  const std::vector<Tensor::value_type> expected = {1, 2, 3, 4};
  auto actual = tensor.storage();
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

  const std::vector<Tensor::value_type> expected = {0, 1, 2, 3, 4, 5, 6, 7};
  auto actual = tensor.storage();
  EXPECT_EQ(actual, expected);
}

// Fill のテスト
TEST(TensorTest, fill)
{
  Tensor tensor = Tensor::Fill({2, 2}, 1);
  EXPECT_EQ(tensor.storage(), (std::vector<Tensor::value_type>{1, 1, 1, 1}));
}

// Fill のテスト (3次元)
TEST(TensorTest, fill3D)
{
  Tensor tensor = Tensor::Fill({2, 2, 2}, 1);
  EXPECT_EQ(tensor.storage(), (std::vector<Tensor::value_type>{1, 1, 1, 1, 1, 1, 1, 1}));
}

// zeros テスト
TEST(TensorTest, zeros)
{
  Tensor tensor = Tensor::Zeros({2, 2});
  EXPECT_EQ(tensor.storage().size(), 4);
  EXPECT_EQ(tensor.storage(), (std::vector<Tensor::value_type>{0, 0, 0, 0}));
}

// ones テスト
TEST(TensorTest, ones)
{
  Tensor tensor = Tensor::Ones({2, 2});
  EXPECT_EQ(tensor.storage().size(), 4);
  EXPECT_EQ(tensor.storage(), (std::vector<Tensor::value_type>{1, 1, 1, 1}));
}

// ones テスト (3次元)
TEST(TensorTest, ones3D)
{
  Tensor tensor = Tensor::Ones({2, 2, 2});
  EXPECT_EQ(tensor.storage().size(), 8);
  EXPECT_EQ(tensor.storage(), (std::vector<Tensor::value_type>{1, 1, 1, 1, 1, 1, 1, 1}));
}

// eye テスト
TEST(TensorTest, eye)
{
  Tensor tensor = Tensor::Eye({2, 2});
  EXPECT_EQ(tensor.storage().size(), 4);
  EXPECT_EQ(tensor.storage(), (std::vector<Tensor::value_type>{1, 0, 0, 1}));
}

// 加算テスト
TEST(TensorTest, add)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a + b;
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{2, 2, 2, 2}));
}

// 減算テスト
TEST(TensorTest, sub)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a - b;
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{0, 0, 0, 0}));
}

// 乗算テスト
TEST(TensorTest, mul)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a * b;
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{1, 1, 1, 1}));
}

// 除算テスト
TEST(TensorTest, div)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = a / b;

  for (std::size_t i = 0; i < c.storage().size(); ++i)
  {
    // 誤差を許容
    EXPECT_NEAR(c.storage()[i], 1.0f, 1e-5);
  }
}

// reshape テスト
TEST(TensorTest, reshape)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = a.Reshape({4});
  EXPECT_EQ(b.storage(), (std::vector<Tensor::value_type>{1, 1, 1, 1}));

  // 新しい形状が期待したものと同じかチェック
  EXPECT_EQ(b.shape(), (std::vector<std::size_t>{4}));
}

// reshape テスト (3次元)
TEST(TensorTest, reshape3D)
{
  Tensor a = Tensor::Ones({2, 2, 2});
  Tensor b = a.Reshape({4, 2});
  EXPECT_EQ(b.storage(), (std::vector<Tensor::value_type>{1, 1, 1, 1, 1, 1, 1, 1}));

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
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{2}));
}

// 2次元のテンソル同士
TEST(TensorTest, dot2D)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor b = Tensor::Ones({2, 2});
  Tensor c = Tensor::Dot(a, b);
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{2, 2}));
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
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{2}));
}

// sum テスト (2次元)
TEST(TensorTest, sum2D)
{
  Tensor a = Tensor::Ones({2, 2});
  Tensor c = Tensor::Sum(a);
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{2, 2}));
}

// sum テスト (3次元)
TEST(TensorTest, sum3D)
{
  Tensor a = Tensor::Ones({2, 2, 2});
  Tensor c = Tensor::Sum(a);
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{2, 2, 2, 2}));
}

// Softmax テスト (1次元)
TEST(TensorTest, softmax)
{
  Tensor a = Tensor::Zeros({3});
  a(0) = 0.3;
  a(1) = 2.9;
  a(2) = 4.0;

  Tensor c = Tensor::Softmax(a);

  // 誤差を考慮して許容範囲を設定
  EXPECT_NEAR(c.storage()[0], 0.01821127, 1e-6);
  EXPECT_NEAR(c.storage()[1], 0.24519181, 1e-6);
  EXPECT_NEAR(c.storage()[2], 0.73659691, 1e-6);
}

// Softmax テスト (2次元)
TEST(TensorTest, softmax2D)
{
  Tensor a = Tensor::Zeros({2, 3});
  a(0, 0) = 0.3;
  a(0, 1) = 2.9;
  a(0, 2) = 4.0;
  a(1, 0) = 0.3;
  a(1, 1) = 2.9;
  a(1, 2) = 4.0;

  Tensor c = Tensor::Softmax(a);

  // 誤差を考慮して許容範囲を設定
  EXPECT_NEAR(c(0, 0), 0.01821127, 1e-6);
  EXPECT_NEAR(c(0, 1), 0.24519181, 1e-6);
  EXPECT_NEAR(c(0, 2), 0.73659691, 1e-6);
  EXPECT_NEAR(c(1, 0), 0.01821127, 1e-6);
  EXPECT_NEAR(c(1, 1), 0.24519181, 1e-6);
  EXPECT_NEAR(c(1, 2), 0.73659691, 1e-6);
}

// Transpose テスト (2次元)
TEST(TensorTest, transpose2D)
{
  Tensor a = Tensor::FromArray({{1, 2, 3}, {4, 5, 6}});
  Tensor c = Tensor::Transpose(a);

  // 転置前の形状をチェック
  EXPECT_EQ(a.shape(), (std::vector<std::size_t>{2, 3}));

  // 転置後の形状をチェック
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{3, 2}));

  Tensor ans = Tensor::FromArray({{1, 4}, {2, 5}, {3, 6}});
  EXPECT_EQ(c.storage(), ans.storage());
}

// Transpose テスト (3次元)
TEST(TensorTest, transpose3D)
{
  Tensor a = Tensor::FromArray({
    {
      {1, 2, 3},
      {4, 5, 6}
    },
    {
      {7, 8, 9},
      {10, 11, 12}
    }
  });
  // 転置前の形状をチェック
  EXPECT_EQ(a.shape(), (std::vector<std::size_t>{2, 2, 3}));

  Tensor c = Tensor::Transpose(a);
  // 転置後の形状をチェック
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 3, 2}));

  Tensor ans = Tensor::FromArray({
    {{1, 4}, {2, 5}, {3, 6}},
    {{7, 10}, {8, 11}, {9, 12}}
  });
  EXPECT_EQ(c.storage(), ans.storage());
}

// Slice テスト
TEST(TensorTest, slice)
{
  Tensor a = Tensor::FromArray({{1, 2, 3}, {4, 5, 6}});
  Tensor c = a.Slice(0);
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{1, 2, 3}));

  // shape が (1, 3) であるかチェック
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{3}));
}

// Slice テスト (3次元)
TEST(TensorTest, slice3D)
{
  Tensor a = Tensor::FromArray(
    {
      {
        {1, 2, 3},
        {4, 5, 6}
      },
      {
        {7, 8, 9},
        {10, 11, 12}
      }
    }
  );
  Tensor c = a.Slice(0);

  // shapeが(2, 3)であるかチェック
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 3}));

  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{1, 2, 3, 4, 5, 6}));
}

// 指定範囲を一括して取得する Slice テスト
TEST(TensorTest, sliceRange)
{
  Tensor a = Tensor::FromArray({
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {10, 11, 12}
  });
  Tensor c = a.Slice(1, 3);

  // shape が (2, 3) であるかチェック
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{3, 3}));

  // 期待される値をチェック
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

// 指定範囲を一括して取得する Slice テスト (3次元)
TEST(TensorTest, sliceRange3D)
{
  Tensor a = Tensor::FromArray({
    {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9}
    },
    {
      {10, 11, 12},
      {13, 14, 15},
      {16, 17, 18}
    },
    {
      {19, 20, 21},
      {22, 23, 24},
      {25, 26, 27}
    },
    {
      {28, 29, 30},
      {31, 32, 33},
      {34, 35, 36}
    }
  });
  Tensor c = a.Slice(1, 2);

  // shape が (2, 3) であるかチェック
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 3, 3}));

  std::vector<Tensor::value_type> expected;
  for (int i = 10; i < 28; i++)
  {
    expected.push_back(i);
  }

  // 期待される値をチェック
  EXPECT_EQ(c.storage(), expected);
}

TEST(TensorTest, NumericalGradient)
{
  // 1次元テンソル [3, 4] を作成
  Tensor x = Tensor::FromArray({3.0f, 4.0f});

  // 関数 f(x) = sum(x^2) を定義する
  auto f = [](const Tensor &t) -> Tensor
  {
    // 各要素の2乗を計算し、その合計（1要素のテンソル）を返す
    Tensor squared = t * t; // 要素ごとの乗算
    Tensor s = Tensor::Sum(squared);
    return s;
  };

  // 数値微分により勾配を算出する
  Tensor grad = numerical_gradient(f, x);

  // 期待される勾配は 2*x なので、[3,4] -> [6,8]
  Tensor expected = Tensor::FromArray({6.0f, 8.0f});

  // 計算結果と期待値を許容誤差 1e-4 で比較
  const auto &grad_storage = grad.storage();
  const auto &expected_storage = expected.storage();
  ASSERT_EQ(grad_storage.size(), expected_storage.size());
  for (std::size_t i = 0; i < grad_storage.size(); ++i)
  {
    EXPECT_NEAR(grad_storage[i], expected_storage[i], 1e-1f);
  }
}

TEST(TensorTest, NumericalGradientBatch)
{
  // 2次元テンソルを作成 (バッチサイズ 2, 特徴量 2)
  Tensor x = Tensor::FromArray({{3.0f, 4.0f}, {5.0f, 6.0f}});

  // 関数 f(x) = sum(x^2) を定義する
  // バッチ内の全要素の二乗和を求め、スカラー値を返す
  auto f = [](const Tensor &t) -> Tensor
  {
    Tensor squared = t * t; // 各要素の2乗
    Tensor s = Tensor::Sum(squared); // 各サンプルごとの和 (shape: {batch})
    s = Tensor::Sum(s); // バッチ全体の和 (スカラー)
    return s;
  };

  // 数値微分により勾配を算出する
  Tensor grad = numerical_gradient(f, x);

  // 期待される勾配は d/dx (x^2) = 2*x なので、[[3,4], [5,6]] に対しては [[6,8], [10,12]] が期待される
  Tensor expected = Tensor::FromArray({{6.0f, 8.0f}, {10.0f, 12.0f}});

  // 計算結果と期待値を許容誤差 1e-2 で比較
  const auto &grad_storage = grad.storage();
  const auto &expected_storage = expected.storage();
  ASSERT_EQ(grad_storage.size(), expected_storage.size());
  for (std::size_t i = 0; i < grad_storage.size(); ++i)
  {
    EXPECT_NEAR(grad_storage[i], expected_storage[i], 1e-1f);
  }
}


// Concat テスト
TEST(TensorTest, concat)
{
  Tensor a = Tensor::FromArray({1, 2, 3});
  Tensor b = Tensor::FromArray({4, 5, 6});
  Tensor c = Tensor::Concat(a, b);

  // 期待される形状は (2, 3) であるかチェック
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 3}));

  // 期待される値をチェック
  EXPECT_EQ(c.storage(), (std::vector<Tensor::value_type>{1, 2, 3, 4, 5, 6}));
}

// Concat テスト (3次元)
TEST(TensorTest, concat3D)
{
  Tensor a = Tensor::FromArray({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
  Tensor b = Tensor::FromArray({{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}});
  Tensor c = Tensor::FromArray({{{25, 26, 27}, {28, 29, 30}}, {{31, 32, 33}, {34, 35, 36}}});
  Tensor d = Tensor::Concat(a, b, c);


  // 期待される形状は (2, 2, 3) であるかチェック
  EXPECT_EQ(d.shape(), (std::vector<std::size_t>{3, 2, 2, 3}));

  // 期待される値をチェック
  EXPECT_EQ(d.storage(), (std::vector<Tensor::value_type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}));
}

// Concat の vector 版テスト (3次元)
TEST(TensorTest, concatVector)
{
  Tensor a = Tensor::FromArray({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
  Tensor b = Tensor::FromArray({{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}});
  Tensor c = Tensor::FromArray({{{25, 26, 27}, {28, 29, 30}}, {{31, 32, 33}, {34, 35, 36}}});
  Tensor d = Tensor::Concat(std::vector{a, b, c});

  // 期待される形状は (2, 2, 3) であるかチェック
  EXPECT_EQ(d.shape(), (std::vector<std::size_t>{3, 2, 2, 3}));

  // 期待される値をチェック
  EXPECT_EQ(d.storage(), (std::vector<Tensor::value_type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}));
}

TEST(SoftmaxWithLossTest, Forward)
{
  // サンプル (batch=2, classes=3) の入力テンソルと教師ラベル (one-hot) を作成
  // サンプル1: x = [0.3, 2.9, 4.0], 正解はクラス2 (one-hot: [0, 0, 1])
  // サンプル2: x = [0.1, 0.2, 0.7], 正解はクラス0 (one-hot: [1, 0, 0])
  Tensor x = Tensor::FromArray({{0.3f, 2.9f, 4.0f},
                                {0.1f, 0.2f, 0.7f}});
  Tensor t = Tensor::FromArray({{0.0f, 0.0f, 1.0f},
                                {1.0f, 0.0f, 0.0f}});

  // SoftmaxWithLoss レイヤーの forward を実行
  SoftmaxWithLoss layer;
  Tensor loss = layer.forward(x, t);

  // 期待する loss を計算すると
  // サンプル1: softmax(x) ≒ [0.0182, 0.2453, 0.7365] → -log(0.7365) ≒ 0.306
  // サンプル2: softmax(x) ≒ [0.2546, 0.2814, 0.4639] → -log(0.2546) ≒ 1.366
  // 平均 loss = (0.306 + 1.366) / 2 ≒ 0.836
  Tensor::value_type loss_value = loss(0);
  EXPECT_NEAR(loss_value, 0.836f, 0.05f);
}


TEST(SoftmaxWithLossTest, Backward)
{
    // forward で使用する同じ入力テンソルと教師ラベル
    Tensor x = Tensor::FromArray({{0.3f, 2.9f, 4.0f},
                                  {0.1f, 0.2f, 0.7f}});
    Tensor t = Tensor::FromArray({{0.0f, 0.0f, 1.0f},
                                  {1.0f, 0.0f, 0.0f}});
    
    SoftmaxWithLoss layer;
    // forward を実行して内部状態 (softmax 結果 y と教師 t) をセット
    layer.forward(x, t);
    
    // backward の入力 dout は通常 1 を与えますが、ここでは内部で
    // dx = (y - t) / batch_size を計算しているので、ダミーとして Ones を渡す
    Tensor dout = Tensor::Ones({1});
    Tensor dx = layer.backward(dout);

    // 手計算による期待される softmax 結果は以下の通り (概算)
    // サンプル1: x = [0.3, 2.9, 4.0]
    //   row_max = 4.0, shifted = [-3.7, -1.1, 0]
    //   exp = [exp(-3.7) ≒ 0.02473, exp(-1.1) ≒ 0.33287, exp(0) = 1.0]
    //   sum = 1.35760 → softmax ≒ [0.0182, 0.2453, 0.7365]
    //   y - t = [0.0182 - 0, 0.2453 - 0, 0.7365 - 1] = [0.0182, 0.2453, -0.2635]
    //   dx (1サンプルあたり) = [0.0091, 0.12265, -0.13175]   (batch_size = 2 で割る)
    //
    // サンプル2: x = [0.1, 0.2, 0.7]
    //   row_max = 0.7, shifted = [-0.6, -0.5, 0]
    //   exp = [exp(-0.6) ≒ 0.54881, exp(-0.5) ≒ 0.60653, exp(0) = 1.0]
    //   sum = 2.15534 → softmax ≒ [0.2546, 0.2814, 0.4639]
    //   y - t = [0.2546 - 1, 0.2814 - 0, 0.4639 - 0] = [-0.7454, 0.2814, 0.4639]
    //   dx = [-0.3727, 0.1407, 0.23195]
    //
    // 期待される dx は下記の 2x3 テンソル
    Tensor expected_dx = Tensor::FromArray({{0.0091f, 0.12265f, -0.13175f},
                                            {-0.3727f, 0.1407f, 0.23195f}});
    
    // dx の shape, 各要素が expected_dx と近似しているか確認
    ASSERT_EQ(dx.shape(), expected_dx.shape());
    const auto &dx_storage = dx.storage();
    const auto &expected_storage = expected_dx.storage();
    ASSERT_EQ(dx_storage.size(), expected_storage.size());
    for (std::size_t i = 0; i < dx_storage.size(); ++i)
    {
        EXPECT_NEAR(dx_storage[i], expected_storage[i], 0.01f);
    }
}

TEST(AffineLayerTest, ForwardBatch) {
    // バッチサイズ:2, 入力次元:2, 出力次元:3
    // 入力テンソル x
    Tensor x = Tensor::FromArray({{1.0f, 2.0f},
                                  {3.0f, 4.0f}});
    // 重み W (shape: 2 x 3) と バイアス b (shape: 1 x 3)
    Tensor W = Tensor::FromArray({{0.1f, 0.2f, 0.3f},
                                  {0.4f, 0.5f, 0.6f}});
    Tensor b = Tensor::FromArray({0.1f, 0.2f, 0.3f}).Reshape({1, 3});

    auto W_ptr = std::make_shared<Tensor>(W);
    auto b_ptr = std::make_shared<Tensor>(b);
    
    // Affine レイヤーのインスタンス生成
    Affine layer(W_ptr, b_ptr);
    
    // forward 実行: output = x * W + b
    Tensor out = layer.forward(x);
    
    // 期待する出力の計算
    // サンプル 1: x = [1, 2]
    //   out = [1*0.1 + 2*0.4 + 0.1, 1*0.2 + 2*0.5 + 0.2, 1*0.3 + 2*0.6 + 0.3]
    //       = [0.1 + 0.8 + 0.1, 0.2 + 1.0 + 0.2, 0.3 + 1.2 + 0.3]
    //       = [1.0, 1.4, 1.8]
    // サンプル 2: x = [3, 4]
    //   out = [3*0.1 + 4*0.4 + 0.1, 3*0.2 + 4*0.5 + 0.2, 3*0.3 + 4*0.6 + 0.3]
    //       = [0.3 + 1.6 + 0.1, 0.6 + 2.0 + 0.2, 0.9 + 2.4 + 0.3]
    //       = [2.0, 2.8, 3.6]
    Tensor expected = Tensor::FromArray({{1.0f, 1.4f, 1.8f},
                                         {2.0f, 2.8f, 3.6f}});
    
    const auto &out_storage = out.storage();
    const auto &expected_storage = expected.storage();
    ASSERT_EQ(out_storage.size(), expected_storage.size());
    for (size_t i = 0; i < out_storage.size(); ++i) {
        EXPECT_NEAR(out_storage[i], expected_storage[i], 1e-5);
    }
}

TEST(AffineLayerTest, BackwardBatch) {
    // forward と同じ設定で初期化
    Tensor x = Tensor::FromArray({{1.0f, 2.0f},
                                  {3.0f, 4.0f}});
    Tensor W = Tensor::FromArray({{0.1f, 0.2f, 0.3f},
                                  {0.4f, 0.5f, 0.6f}});
    Tensor b = Tensor::FromArray({0.1f, 0.2f, 0.3f}).Reshape({1, 3});
    
    auto W_ptr = std::make_shared<Tensor>(W);
    auto b_ptr = std::make_shared<Tensor>(b);
    
    Affine layer(W_ptr, b_ptr);
    // forward pass を実行（内部で x を記憶）
    Tensor out = layer.forward(x);
    
    // dout を定義 (shape: 2 x 3)
    // 例として、以下のような勾配をアップストリームから受け取ったとする
    Tensor dout = Tensor::FromArray({{1.0f, 2.0f, 3.0f},
                                     {4.0f, 5.0f, 6.0f}});
    
    // backward 実行 (dx を返し、内部で dW, db を計算)
    Tensor dx = layer.backward(dout);
    
    // 期待される dW の計算: dW = x^T * dout
    // x^T = [[1, 3],
    //        [2, 4]]
    // dout = [[1, 2, 3],
    //         [4, 5, 6]]
    // dW[0,0] = 1*1 + 3*4 = 13,  dW[0,1] = 1*2 + 3*5 = 17,  dW[0,2] = 1*3 + 3*6 = 21
    // dW[1,0] = 2*1 + 4*4 = 18,  dW[1,1] = 2*2 + 4*5 = 24,  dW[1,2] = 2*3 + 4*6 = 30
    Tensor expected_dW = Tensor::FromArray({{13.0f, 17.0f, 21.0f},
                                              {18.0f, 24.0f, 30.0f}});
    
    // 期待される db の計算: db = Tensor::Sum(dout, 0)
    // 各列の和を計算 → [1+4, 2+5, 3+6] = [5, 7, 9]
    Tensor expected_db = Tensor::FromArray({5.0f, 7.0f, 9.0f}).Reshape({1, 3});
    
    // 期待される dx の計算: dx = dout * (W^T)
    // W = [[0.1, 0.2, 0.3],
    //      [0.4, 0.5, 0.6]]
    // W^T = [[0.1, 0.4],
    //        [0.2, 0.5],
    //        [0.3, 0.6]]
    // dx[0,0] = 1*0.1 + 2*0.2 + 3*0.3 = 1.4,  dx[0,1] = 1*0.4 + 2*0.5 + 3*0.6 = 3.2
    // dx[1,0] = 4*0.1 + 5*0.2 + 6*0.3 = 3.2,  dx[1,1] = 4*0.4 + 5*0.5 + 6*0.6 = 7.7
    Tensor expected_dx = Tensor::FromArray({{1.4f, 3.2f},
                                             {3.2f, 7.7f}});
    
    // layer 内部で計算された dW, db の取得
    Tensor calculated_dW = layer.get_dW();
    Tensor calculated_db = layer.get_db();
    
    // dx の検証
    const auto &dx_storage = dx.storage();
    const auto &expected_dx_storage = expected_dx.storage();
    ASSERT_EQ(dx_storage.size(), expected_dx_storage.size());
    for (size_t i = 0; i < dx_storage.size(); ++i) {
        EXPECT_NEAR(dx_storage[i], expected_dx_storage[i], 1e-5);
    }
    
    // dW の検証
    const auto &dW_storage = calculated_dW.storage();
    const auto &expected_dW_storage = expected_dW.storage();
    ASSERT_EQ(dW_storage.size(), expected_dW_storage.size());
    for (size_t i = 0; i < dW_storage.size(); ++i) {
        EXPECT_NEAR(dW_storage[i], expected_dW_storage[i], 1e-5);
    }
    
    // db の検証
    const auto &db_storage = calculated_db.storage();
    const auto &expected_db_storage = expected_db.storage();
    ASSERT_EQ(db_storage.size(), expected_db_storage.size());
    for (size_t i = 0; i < db_storage.size(); ++i) {
        EXPECT_NEAR(db_storage[i], expected_db_storage[i], 1e-5);
    }
}

TEST(ReLUTest, Forward) {
  // 入力テンソル (2x3) を作成。負の値を含む。
  Tensor x = Tensor::FromArray({{-1.0f, 0.0f, 1.0f},
                                {3.0f, -2.0f, 4.0f}});
  // 期待される出力：負の値は 0 に変換される
  Tensor expected = Tensor::FromArray({{0.0f, 0.0f, 1.0f},
                                       {3.0f, 0.0f, 4.0f}});
  
  ReLU relu;
  Tensor out = relu.forward(x);
  
  ASSERT_EQ(out.shape(), expected.shape());
  const auto &out_storage = out.storage();
  const auto &expected_storage = expected.storage();
  ASSERT_EQ(out_storage.size(), expected_storage.size());
  for (size_t i = 0; i < out_storage.size(); ++i) {
    EXPECT_NEAR(out_storage[i], expected_storage[i], 1e-5);
  }
}

TEST(ReLUTest, Backward) {
    // 負の値、0、正の値を含む入力テンソル
    Tensor x = Tensor::FromArray({-1.0f, 0.0f, 2.0f, 0.0f, 3.0f});
    
    // ReLU レイヤーの生成
    ReLU relu;
    
    // 順伝播（内部に x を保持）
    Tensor out = relu.forward(x);
    
    // dout は任意の値（ここでは各要素異なる値を与える）
    Tensor dout = Tensor::FromArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    
    // backward 実行
    Tensor dx = relu.backward(dout);
    
    // 期待される勾配
    //  x[0] = -1.0 （負）→ 勾配は 0
    //  x[1] =  0.0 （入力が 0 の場合は 0 とする）→ 勾配は 0
    //  x[2] =  2.0 （正）→ 勾配は dout[2] = 3.0
    //  x[3] =  0.0 （入力が 0 の場合は 0 とする）→ 勾配は 0
    //  x[4] =  3.0 （正）→ 勾配は dout[4] = 5.0
    Tensor expected = Tensor::FromArray({0.0f, 0.0f, 3.0f, 0.0f, 5.0f});
    
    const auto& dx_storage = dx.storage();
    const auto& expected_storage = expected.storage();
    ASSERT_EQ(dx_storage.size(), expected_storage.size());
    for (std::size_t i = 0; i < dx_storage.size(); ++i) {
        EXPECT_NEAR(dx_storage[i], expected_storage[i], 1e-5);
    }
}

// Reshape のテスト: 元の値が保持され、shape の変換が正しく行われるか検証するテスト
TEST(TensorTest, ReshapePreservesData) {
  // 2x2 のテンソルを作成し、1次元への reshape を行う
  Tensor a = Tensor::FromArray({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Tensor flat = a.Reshape({4});
  EXPECT_EQ(flat.shape(), (std::vector<std::size_t>{4}));
  EXPECT_EQ(flat.storage(), (std::vector<Tensor::value_type>{1.0f, 2.0f, 3.0f, 4.0f}));

  // flat テンソルを元の 2x2 に再度 reshape し、ストレージが変わらないことを確認
  Tensor a2 = flat.Reshape({2, 2});
  EXPECT_EQ(a2.shape(), (std::vector<std::size_t>{2, 2}));
  EXPECT_EQ(a2.storage(), a.storage());
}

TEST(TensorTest, ReshapeMultipleDimensions) {
  // 3次元テンソル (2 x 3 x 4) を 2次元テンソル (4 x 6) に reshape するテスト
  Tensor a = Tensor::Ones({2, 3, 4});
  Tensor b = a.Reshape({4, 6});
  EXPECT_EQ(b.shape(), (std::vector<std::size_t>{4, 6}));
  EXPECT_EQ(b.storage().size(), 24);
  // 全要素が 1.0f であることをチェック
  for (auto v : b.storage()) {
    EXPECT_NEAR(v, 1.0f, 1e-5);
  }
}

// 数学関数のテスト: Exp 関数による各要素の指数計算
TEST(TensorTest, ExpFunction) {
  // 入力テンソルの各要素に exp を適用
  Tensor a = Tensor::FromArray({0.0f, 1.0f, 2.0f});
  Tensor b = Tensor::Exp(a);
  // 期待値: exp(0)=1, exp(1)≒2.71828, exp(2)≒7.38906
  EXPECT_NEAR(b(0), 1.0f, 1e-5);
  EXPECT_NEAR(b(1), 2.71828f, 1e-5);
  EXPECT_NEAR(b(2), 7.38906f, 1e-5);
}

// 数学関数のテスト: Log 関数による各要素の自然対数計算
TEST(TensorTest, LogFunction) {
  // 対数の定義域内の値でテンソルを作成
  Tensor a = Tensor::FromArray({1.0f, 2.71828f, 7.38906f});
  Tensor b = Tensor::Log(a);
  // 期待値: log(1)=0, log(2.71828)≒1, log(7.38906)≒2
  EXPECT_NEAR(b(0), 0.0f, 1e-5);
  EXPECT_NEAR(b(1), 1.0f, 1e-5);
  EXPECT_NEAR(b(2), 2.0f, 1e-5);
}

// ブロードキャストのテスト: 異なる形状のテンソル間の加算
TEST(TensorTest, BroadcastAdd) {
  // a の shape: (2,3), b の shape: (1,3) -> b が各行にブロードキャストされる想定
  Tensor a = Tensor::FromArray({{1.0f, 2.0f, 3.0f},
                                {4.0f, 5.0f, 6.0f}});
  Tensor b = Tensor::FromArray({10.0f, 20.0f, 30.0f}).Reshape({1, 3});
  Tensor c = a + b;
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 3}));
  std::vector<Tensor::value_type> expected = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
  EXPECT_EQ(c.storage(), expected);
}

// ブロードキャストのテスト: 異なる形状のテンソル間の要素ごとの積
TEST(TensorTest, BroadcastMultiply) {
  // a の shape: (2,3), b の shape: (2,1) -> b が各列にブロードキャストされる想定
  Tensor a = Tensor::FromArray({{1.0f, 2.0f, 3.0f},
                                {4.0f, 5.0f, 6.0f}});
  Tensor b = Tensor::FromArray({2.0f, 3.0f}).Reshape({2, 1});
  Tensor c = a * b;
  EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 3}));
  std::vector<Tensor::value_type> expected = {2.0f, 4.0f, 6.0f, 12.0f, 15.0f, 18.0f};
  EXPECT_EQ(c.storage(), expected);
}

// 数値微分を実装するヘルパー関数
Tensor numerical_gradient(std::function<Tensor::value_type(const Tensor&)> f, const Tensor& x) {
    // x と同じ shape のゼロテンソルを生成
    Tensor grad = Tensor::Zeros(x.shape());
    constexpr Tensor::value_type h = 1e-4;
    Tensor x_copy = x;  // x のコピー（深いコピーが行われる前提）
    auto &x_storage = x_copy.storage();
    auto &grad_storage = grad.storage();
    for (size_t i = 0; i < x_storage.size(); ++i) {
        Tensor::value_type temp = x_storage[i];
        x_storage[i] = temp + h;
        Tensor::value_type fxh1 = f(x_copy);
        x_storage[i] = temp - h;
        Tensor::value_type fxh2 = f(x_copy);
        grad_storage[i] = (fxh1 - fxh2) / (2.0f * h);
        x_storage[i] = temp;  // 値を元に戻す
    }
    return grad;
}

// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
// ReLU レイヤーの数値微分による勾配検証テスト
// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
TEST(ReLULayer, NumericalGradient) {
    // 入力テンソル（負の値と正の値を含む）
    Tensor x = Tensor::FromArray({-1.0f, 2.0f, -3.0f, 4.0f});
    
    // ReLU レイヤーのインスタンス（forward(x) で mask を内部に保持し、出力を返す実装とする）
    ReLU relu;
    
    // loss = sum(ReLU(x)) と定義
    auto f = [&](const Tensor& x_var) -> Tensor::value_type {
        // テストごとに新たなインスタンスを作成することで、内部状態の影響を排除
        ReLU relu_temp;
        Tensor y = relu_temp.forward(x_var);
        Tensor::value_type loss = 0.0f;
        for (auto v : y.storage())
            loss += v;
        return loss;
    };
    
    // 数値的勾配を計算
    Tensor num_grad = numerical_gradient(f, x);
    
    // 解析的勾配の計算
    // loss = sum(y) の場合、dout はすべて 1 のテンソルとなる
    Tensor y = relu.forward(x);  // forward 実行して内部に mask を保持させる
    Tensor dout = Tensor::Ones(y.shape());
    Tensor anal_grad = relu.backward(dout);
    
    // 数値微分と解析的勾配の各要素を比較
    const auto &num_storage = num_grad.storage();
    const auto &anal_storage = anal_grad.storage();
    ASSERT_EQ(num_storage.size(), anal_storage.size());
    for (size_t i = 0; i < num_storage.size(); ++i) {
        EXPECT_NEAR(num_storage[i], anal_storage[i], 1e-2)
            << "Mismatch at index " << i;
    }
}

// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
// Affine レイヤーの入力 x に対する数値微分による勾配検証テスト
// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
TEST(AffineLayer, NumericalGradientInput) {
    // 入力テンソル x: バッチサイズ 2, 入力次元 2
    Tensor x = Tensor::FromArray({{1.0f, 2.0f},
                                  {3.0f, 4.0f}});
    // 重み W: 入力次元 2, 出力次元 3 および バイアス b: shape (1, 3)
    Tensor W = Tensor::FromArray({{0.1f, 0.2f, 0.3f},
                                  {0.4f, 0.5f, 0.6f}});
    Tensor b = Tensor::FromArray({0.1f, 0.2f, 0.3f}).Reshape({1, 3});
    
    // Affine レイヤーの生成（パラメータは shared_ptr 経由で与える）
    auto W_ptr = std::make_shared<Tensor>(W);
    auto b_ptr = std::make_shared<Tensor>(b);
    Affine affine_layer(W_ptr, b_ptr);
    
    // loss = sum(affine_layer.forward(x)) と定義（パラメータは固定）
    auto f = [&](const Tensor &x_var) -> Tensor::value_type {
        // 新たな Affine インスタンスを用いて計算（内部状態の干渉を避ける）
        Affine affine_tmp(W_ptr, b_ptr);
        Tensor out = affine_tmp.forward(x_var);
        Tensor::value_type loss = 0.0f;
        for (auto v : out.storage())
            loss += v;
        return loss;
    };
    
    // 数値的勾配（入力xに対する勾配）を計算
    Tensor num_grad = numerical_gradient(f, x);
    
    // 解析的勾配の計算
    Tensor out = affine_layer.forward(x);  // forward で内部に x を保持
    Tensor dout = Tensor::Ones(out.shape());
    Tensor anal_grad = affine_layer.backward(dout);
    
    // 数値微分と解析的勾配を比較
    const auto &num_storage = num_grad.storage();
    const auto &anal_storage = anal_grad.storage();
    ASSERT_EQ(num_storage.size(), anal_storage.size());
    for (size_t i = 0; i < num_storage.size(); ++i) {
        EXPECT_NEAR(num_storage[i], anal_storage[i], 1e-1)
            << "Mismatch at index " << i;
    }
}

// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
// Affine レイヤーの重み W に対する数値微分による勾配検証テスト
// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
TEST(AffineLayer, NumericalGradientWeight) {
    // 入力テンソル x: バッチサイズ 2, 入力次元 2
    Tensor x = Tensor::FromArray({{1.0f, 2.0f},
                                  {3.0f, 4.0f}});
    // 重み W: 入力次元 2, 出力次元 3 および バイアス b: shape (1, 3)
    Tensor W = Tensor::FromArray({{0.1f, 0.2f, 0.3f},
                                  {0.4f, 0.5f, 0.6f}});
    Tensor b = Tensor::FromArray({0.1f, 0.2f, 0.3f}).Reshape({1, 3});
    
    // Affine レイヤーの生成
    // 数値微分対象となるのは重み W なので、これを変数とする
    Affine affine_layer(std::make_shared<Tensor>(W), std::make_shared<Tensor>(b));
    
    // loss = sum(affine_layer.forward(x)) と定義（ここでパラメータ W を変化させる）
    auto f = [&](const Tensor &W_var) -> Tensor::value_type {
        // W_var を新たなレイヤーに渡して forward を実行
        auto W_ptr_var = std::make_shared<Tensor>(W_var);
        Affine affine_tmp(W_ptr_var, std::make_shared<Tensor>(b));
        Tensor out = affine_tmp.forward(x);
        Tensor::value_type loss = 0.0f;
        for (auto v : out.storage())
            loss += v;
        return loss;
    };
    
    // 数値的勾配（重み W に対する勾配）を計算
    Tensor num_grad = numerical_gradient(f, W);
    
    // 解析的勾配の計算
    // forward を実行して内部に x を保持し、backward により dW を計算する
    Tensor out = affine_layer.forward(x);
    Tensor dout = Tensor::Ones(out.shape());
    affine_layer.backward(dout);
    // affine_layer.dW には解析的な重みの勾配が格納されている（テスト用に公開している前提）
    Tensor anal_grad = affine_layer.get_dW();
    
    // 数値微分と解析的勾配を比較
    const auto &num_storage = num_grad.storage();
    const auto &anal_storage = anal_grad.storage();
    ASSERT_EQ(num_storage.size(), anal_storage.size());
    for (size_t i = 0; i < num_storage.size(); ++i) {
        EXPECT_NEAR(num_storage[i], anal_storage[i], 1e-1)
            << "Mismatch at index " << i;
    }
}

TEST(TwoLayerNetTest, NumericalGradient) {
    // テスト用の単純なデータセット:
    // 入力 x: shape (1,2)
    Tensor x = Tensor::FromArray({1.0f, 2.0f}).Reshape({1, 2});
    // ターゲット t: one-hot 表現 (1,2) 
    // ここでは正解ラベルが 1（2番目のクラス）とする
    Tensor t = Tensor::FromArray({0.0f, 1.0f}).Reshape({1, 2});
    
    // 重み初期化の標準偏差
    Tensor::value_type weight_init_std = 0.1f;
    // ネットワークの構築: 入力サイズ 2, 隠れ層サイズ 3, 出力サイズ 2
    TwoLayerNet net(2, 3, 2, weight_init_std);
    
    // 数値微分による勾配の計算
    auto numerical_grads = net.numerical_gradient(x, t);
    
    // 逆伝播で計算した解析的勾配の取得
    auto analytical_grads = net.gradient(x, t);
    
    // 数値勾配と解析的勾配の各パラメータについて、各要素が近似しているかを確認する
    const Tensor::value_type tolerance = 1e-4f;
    ASSERT_EQ(numerical_grads.size(), analytical_grads.size())
        << "勾配パラメータの数が一致していません。";
    
    for (size_t i = 0; i < numerical_grads.size(); ++i) {
        const auto &param_name = numerical_grads[i].first;
        const auto &num_grad = numerical_grads[i].second;
        const auto &ana_grad = analytical_grads[i].second;
        const auto &num_storage = num_grad.storage();
        const auto &ana_storage = ana_grad.storage();
        ASSERT_EQ(num_storage.size(), ana_storage.size())
            << "パラメータ " << param_name << " の勾配サイズが一致していません。";
        for (size_t j = 0; j < num_storage.size(); ++j) {
            EXPECT_NEAR(num_storage[j], ana_storage[j], tolerance)
                << "パラメータ " << param_name << " の index " << j << " で不一致。";
        }
    }
}


TEST(TrainingLoopTest, LossDecreases) {
    // シンプルなデータセット: 1サンプルの入力（サイズ2）とone-hotターゲット（サイズ2）
    Tensor x = Tensor::FromArray({1.0f, 2.0f}).Reshape({1, 2});
    Tensor t = Tensor::FromArray({0.0f, 1.0f}).Reshape({1, 2});

    // TwoLayerNet の構築: 入力サイズ2, 隠れ層サイズ3, 出力サイズ2, 重み初期化標準偏差0.1
    TwoLayerNet net(2, 3, 2, 0.1f);

    // 学習前の損失を計算（1要素のテンソルと仮定し先頭要素を参照）
    Tensor::value_type initial_loss = net.loss(x, t)(0);

    const int iterations = 1000;
    const Tensor::value_type learning_rate = 0.01f;

    // 学習ループ
    for (int i = 0; i < iterations; ++i) {
        // 逆伝播による解析的勾配を計算
        auto grads = net.gradient(x, t);
        // 各パラメータを勾配の方向に更新
        for (auto &grad_pair : grads) {
            for (auto &param_pair : net.params) {
                if (param_pair.first == grad_pair.first) {
                    // Tensor 同士の演算（スカラー倍・引き算）が定義されている前提
                    *(param_pair.second) = *(param_pair.second) - grad_pair.second * learning_rate;
                }
            }
        }
    }

    // 学習後の損失を計算
    Tensor::value_type final_loss = net.loss(x, t)(0);

    // 損失が減少していることを確認
    EXPECT_LT(final_loss, initial_loss)
        << "学習前の損失 (" << initial_loss << ") と比較して、学習後の損失 (" << final_loss << ") が下がっていません。";
}

TEST(SoftmaxWithLoss, NumericalGradient) {
    // バッチサイズ2、クラス数3のケースを想定
    Tensor x = Tensor::FromArray({{1.0f, 2.0f, 3.0f},
                                  {1.5f, 2.5f, 3.5f}});
    // ターゲット t は one-hot 表現
    Tensor t = Tensor::FromArray({{0.0f, 0.0f, 1.0f},
                                  {0.0f, 1.0f, 0.0f}});
    
    // 数値微分用の lambda を定義（各評価で新たな SoftmaxWithLoss インスタンスを利用）
    auto f = [&](const Tensor &x_var) -> Tensor::value_type {
        SoftmaxWithLoss loss_tmp;
        Tensor loss_val = loss_tmp.forward(x_var, t);
        // loss_val は1要素のテンソルと仮定
        return loss_val(0);
    };

    // 数値微分による勾配を計算
    Tensor num_grad = numerical_gradient(f, x);

    // 解析的勾配の計算
    SoftmaxWithLoss loss_layer;
    Tensor loss_val = loss_layer.forward(x, t);
    Tensor dout = Tensor::Ones(loss_val.shape());
    Tensor ana_grad = loss_layer.backward(dout);

    // 数値勾配と解析的勾配を比較（許容誤差は適宜調整）
    const auto &num_storage = num_grad.storage();
    const auto &ana_storage = ana_grad.storage();
    ASSERT_EQ(num_storage.size(), ana_storage.size());
    for (std::size_t i = 0; i < num_storage.size(); ++i) {
        EXPECT_NEAR(num_storage[i], ana_storage[i], 1e-2)
            << "index " << i << ": 数値勾配 " << num_storage[i]
            << " vs 解析的勾配 " << ana_storage[i];
    }
}
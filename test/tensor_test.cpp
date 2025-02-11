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
    constexpr float h = 1e-3;

    // x と同じ形状を持つゼロ初期化のテンソルを作成する
    Tensor grad = Tensor::Zeros(x.shape());
    // x の変更可能なコピーを作成する
    Tensor x_copy = x;

    // Tensor のストレージ全体（全要素）でループ
    for (std::size_t idx = 0; idx < x_copy.storage().size(); ++idx)
    {
        // 現在の値を記憶
        float tmp_val = x_copy.storage()[idx];

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
  EXPECT_EQ(c.storage(), (std::vector<float>{1, 2, 3}));

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

  EXPECT_EQ(c.storage(), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(TensorTest, NumericalGradient) {
    // 1次元テンソル [3, 4] を作成
    Tensor x = Tensor::FromArray({3.0f, 4.0f});
    
    // 関数 f(x) = sum(x^2) を定義する
    auto f = [](const Tensor &t) -> Tensor {
        // 各要素の2乗を計算し、その合計（1要素のテンソル）を返す
        Tensor squared = t * t;  // 要素ごとの乗算
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
    for (std::size_t i = 0; i < grad_storage.size(); ++i) {
        EXPECT_NEAR(grad_storage[i], expected_storage[i], 1e-1f);
    }
}

TEST(TensorTest, NumericalGradientBatch) {
    // 2次元テンソルを作成 (バッチサイズ 2, 特徴量 2)
    Tensor x = Tensor::FromArray({{3.0f, 4.0f}, {5.0f, 6.0f}});
    
    // 関数 f(x) = sum(x^2) を定義する
    // バッチ内の全要素の二乗和を求め、スカラー値を返す
    auto f = [](const Tensor &t) -> Tensor {
        Tensor squared = t * t;  // 各要素の2乗
        Tensor s = Tensor::Sum(squared);  // 各サンプルごとの和 (shape: {batch})
        s = Tensor::Sum(s);  // バッチ全体の和 (スカラー)
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
    for (std::size_t i = 0; i < grad_storage.size(); ++i) {
        EXPECT_NEAR(grad_storage[i], expected_storage[i], 1e-1f);
    }
}
//
// Created by toru on 2024/09/08.
//
#include <gtest/gtest.h>
#include "nagatolib.hpp"
#include "narray.hpp"
using namespace nagato;

TEST(NArrayTest, NArrayFill)
{
  auto a = na::Fill<float, 4, 4>(1.5f);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      EXPECT_EQ(a(i, j), 1.5f);
    }
  }
}
TEST(NArrayTest, FillMatrix)
{
  using namespace nagato;
  auto a = na::Fill<float, 4, 4>(1.5f);

  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      EXPECT_EQ(a(i, j), 1.5f);
    }
  }
}

TEST(NArrayTest, ArrayAccess)
{
  using namespace nagato;
  auto a = na::Fill<float, 4, 4>(1.5f);

  EXPECT_EQ(a[0][0], 1.5f);

  a[0][0] = 2.f;

  EXPECT_EQ(a[0][0], 2.f);
}

TEST(NArrayTest, CopyArray)
{
  using namespace nagato;
  auto a = na::Fill<float, 4>(1.f);
  auto b = na::Copy(a);

  EXPECT_EQ(b[0], 1.f);
  EXPECT_EQ(b[1], 1.f);

  a[1] = 2.f;
  EXPECT_EQ(b[1], 1.f);  // コピーされた値は変更されない
}

TEST(NArrayTest, AsTypeConversion)
{
  using namespace nagato;
  auto a = na::Fill<float, 4, 4>(1.5f);
  const auto f = na::AsType<int>(a);

  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      EXPECT_EQ(f(i, j), 1);  // 1.5fはintに変換されると1
    }
  }
}

TEST(NArrayTest, SingleElementArray)
{
  using namespace nagato;

  // NagatoArray の作成
  const auto a = na::NagatoArray<int, 1, 1, 1>(1);

  // 正しい初期化が行われているかの確認
  EXPECT_EQ(a(0, 0, 0), 1);
}

TEST(NArrayTest, ArrayTransformAddition)
{
  using namespace nagato;

  // 3x3x3 の NagatoArray を初期化
  const auto a = na::NagatoArray<float, 3, 3, 3>(4);
  const auto b = na::NagatoArray<float, 3, 3, 3>(2);

  // 2つの配列の要素ごとに加算する
  const auto c =
    na::Transform(a,
                  b,
                  [](auto x, auto y)
                  { return x + y; }
    );

  // 各要素が正しく加算されているかをチェック
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        EXPECT_EQ(c(i, j, k), 6.0f);  // 4 + 2 = 6
      }
    }
  }
}

TEST(NArrayTest, TransformSingleArray)
{
  using namespace nagato;

  // 3x3x3 の NagatoArray を初期化
  const auto a = na::NagatoArray<float, 3, 3, 3>(4);

  // 配列の各要素を2倍にするTransformを適用
  const auto c = na::Transform(a,
                               [](auto v)
                               { return v * 2; }
  );

  // 各要素が正しく2倍になっているかを確認
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        EXPECT_EQ(c(i, j, k), 8.0f);  // 4 * 2 = 8
      }
    }
  }
}

TEST(NArrayTest, ArrayAddition)
{
  using namespace nagato;

  // 3x3x3 の NagatoArray を初期化
  const auto a = na::NagatoArray<float, 3, 3, 3>(4);
  const auto b = na::NagatoArray<float, 3, 3, 3>(2);

  // 配列同士の加算
  const auto c = a + b;

  // 配列とスカラーの加算
  const auto d = a + 3;
  const auto e = 4 + a;

  // それぞれの結果をチェック
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        EXPECT_EQ(c(i, j, k), 6.0f);  // 4 + 2 = 6
        EXPECT_EQ(d(i, j, k), 7.0f);  // 4 + 3 = 7
        EXPECT_EQ(e(i, j, k), 8.0f);  // 4 + 4 = 8
      }
    }
  }
}

TEST(NArrayTest, ArraySubtraction)
{
  using namespace nagato;

  // 3x3x3 の NagatoArray を初期化
  const auto a = na::NagatoArray<float, 3, 3, 3>(4);
  const auto b = na::NagatoArray<float, 3, 3, 3>(2);

  // 配列同士の減算
  const auto c = a - b;

  // 配列とスカラーの減算
  const auto d = a - 3;
  const auto e = 4 - a;

  // それぞれの結果をチェック
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        EXPECT_EQ(c(i, j, k), 2.0f);  // 4 - 2 = 2
        EXPECT_EQ(d(i, j, k), 1.0f);  // 4 - 3 = 1
        EXPECT_EQ(e(i, j, k), 0.0f);  // 4 - 4 = 0
      }
    }
  }
}

TEST(NArrayTest, ArrayMultiplication)
{
  using namespace nagato;

  // 3x3x3 の NagatoArray を初期化
  const auto a = na::NagatoArray<float, 3, 3, 3>(4);
  const auto b = na::NagatoArray<float, 3, 3, 3>(2);

  // 配列同士の乗算
  const auto c = a * b;

  // 配列とスカラーの乗算
  const auto d = a * 3;
  const auto e = 4 * a;

  // それぞれの結果をチェック
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        EXPECT_EQ(c(i, j, k), 8.0f);  // 4 * 2 = 8
        EXPECT_EQ(d(i, j, k), 12.0f); // 4 * 3 = 12
        EXPECT_EQ(e(i, j, k), 16.0f); // 4 * 4 = 16
      }
    }
  }
}

TEST(NArrayTest, ArrayDivision)
{
  using namespace nagato;

  // 3x3x3 の NagatoArray を初期化
  const auto a = na::NagatoArray<float, 3, 3, 3>(4);
  const auto b = na::NagatoArray<float, 3, 3, 3>(2);

  // 配列同士の除算
  const auto c = a / b;

  // 配列とスカラーの除算
  const auto d = a / 2;
  const auto e = 8 / a;

  // それぞれの結果をチェック
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 3; ++k)
      {
        EXPECT_EQ(c(i, j, k), 2.0f);  // 4 / 2 = 2
        EXPECT_EQ(d(i, j, k), 2.0f);  // 4 / 2 = 2
        EXPECT_EQ(e(i, j, k), 2.0f);  // 8 / 4 = 2
      }
    }
  }
}

TEST(NArrayTest, ArrayEquality)
{
  using namespace nagato;

  // 3x3x3 の NagatoArray を初期化
  const auto a = na::NagatoArray<float, 3, 3, 3>(4);
  const auto b = na::NagatoArray<float, 3, 3, 3>(2);
  const auto c = na::NagatoArray<float, 3, 3, 3>(4);

  // 配列同士の比較
  EXPECT_FALSE(a == b);  // a と b は異なるため false
  EXPECT_TRUE(a == c);   // a と c は同じため true
}

TEST(NArrayTest, ArrayComparison)
{
  using namespace nagato;

  // 4x4 の NagatoArray を初期化
  auto a = na::Fill<float, 4, 4>(1.5f);
  auto b = na::Fill<float, 4, 4>(2.5f);

  // b[0][0] を変更
  b[0][0] = 1.5f;

  // 配列 a と b の比較
  auto c = a < b;

  // 結果が期待通りか確認
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      if (i == 0 && j == 0)
      {
        EXPECT_FALSE(c(i, j));  // a[0][0] == b[0][0] なので false
      }
      else
      {
        EXPECT_TRUE(c(i, j));   // それ以外の要素は a < b なので true
      }
    }
  }
}

TEST(NArrayTest, GreaterThanComparison)
{
  using namespace nagato;

  // 4x4 の NagatoArray を初期化
  auto a = na::Fill<float, 4, 4>(3.5f);
  auto b = na::Fill<float, 4, 4>(2.5f);

  // b[0][0] を変更
  b[0][0] = 3.5f;

  // 配列 a と b の比較
  auto c = a > b;

  // 結果が期待通りか確認
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      if (i == 0 && j == 0)
      {
        EXPECT_FALSE(c(i, j));  // a[0][0] == b[0][0] なので false
      }
      else
      {
        EXPECT_TRUE(c(i, j));   // a > b なので true
      }
    }
  }
}

TEST(NArrayTest, LessThanOrEqualComparison)
{
  using namespace nagato;

  // 4x4 の NagatoArray を初期化
  auto a = na::Fill<float, 4, 4>(2.5f);
  auto b = na::Fill<float, 4, 4>(2.5f);

  // b[0][0] を変更
  b[0][0] = 3.0f;

  // 配列 a と b の比較
  auto c = a <= b;

  // 結果が期待通りか確認
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      if (i == 0 && j == 0)
      {
        EXPECT_TRUE(c(i, j));  // a[0][0] <= b[0][0] なので true
      }
      else
      {
        EXPECT_TRUE(c(i, j));  // a == b なので true
      }
    }
  }
}

TEST(NArrayTest, GreaterThanOrEqualComparison)
{
  using namespace nagato;

  // 4x4 の NagatoArray を初期化
  auto a = na::Fill<float, 4, 4>(2.5f);
  auto b = na::Fill<float, 4, 4>(2.5f);

  // b[0][0] を変更
  b[0][0] = 2.0f;

  // 配列 a と b の比較
  auto c = a >= b;

  // 結果が期待通りか確認
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      if (i == 0 && j == 0)
      {
        EXPECT_TRUE(c(i, j));  // a[0][0] >= b[0][0] なので true
      }
      else
      {
        EXPECT_TRUE(c(i, j));  // a == b なので true
      }
    }
  }
}

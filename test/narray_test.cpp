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

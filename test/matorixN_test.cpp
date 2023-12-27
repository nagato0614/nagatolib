//
// Created by toru on 2023/12/28.
//
#include <gtest/gtest.h>
#include "matrix_n.hpp"

using namespace nagato;

TEST(MatrixNTest, MatrixNAddition)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };
  auto c = a + b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j] + b[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixNSubtraction)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };
  auto c = a - b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j] - b[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixNMultiplication)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };

  auto c  = a * b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(c[i][j], a[i][j] * b[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixNDivision)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };

  auto c = a / b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j] / b[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixNAdditionAssignment)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };
  a += b;

  MatrixN c = {
    {6, 8},
    {10, 12},
  };
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixNSubtractionAssignment)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };
  MatrixN<float> c = {
    {-4, -4},
    {-4, -4},
  };
  a -= b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixNMultiplicationAssignment)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };
  a *= b;
  MatrixN<float> c = {
    {5, 12},
    {21, 32},
  };
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixNDivisionAssignment)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };

  a /= b;
  MatrixN<float> c = {
    {1.0f / 5.0f, 2.0f / 6.0f},
    {3.0f / 7.0f, 4.0f / 8.0f},
  };
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixDot)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> b = {
    {5, 6},
    {7, 8},
  };
  MatrixN<float> c = {
    {19, 22},
    {43, 50},
  };

  auto d = Dot(a, b);

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(d[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixDot_23_32)
{
  MatrixN<float> a = {
    {1, 2, 3},
    {4, 5, 6},
  };
  MatrixN<float> b = {
    {1, 2},
    {3, 4},
    {5, 6},
  };
  MatrixN<float> c = {
    {22, 28},
    {49, 64},
  };

  auto d = Dot(a, b);
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(d[i][j], c[i][j]);
    }
  }
}
TEST(MatrixNTest, MatrixDot_32_23)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
    {5, 6},
  };
  MatrixN<float> b = {
    {1, 2, 3},
    {4, 5, 6},
  };
  MatrixN<float> c = {
    {9, 12, 15},
    {19, 26, 33},
    {29, 40, 51},
  };

  auto d = Dot(a, b);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++) {
      EXPECT_EQ(d[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixDot_33_33)
{
  MatrixN<float> a = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
  };
  MatrixN<float> b = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
  };
  MatrixN<float> c = {
    {30, 36, 42},
    {66, 81, 96},
    {102, 126, 150},
  };

  auto d = Dot(a, b);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++) {
      EXPECT_EQ(d[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixDot_32_34)
{
  MatrixN<float> a = {
    {1, 2},
    {3, 4},
    {5, 6},
  };
  MatrixN<float> b = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
  };
  MatrixN<float> c = {
    {11, 14, 17, 20},
    {23, 30, 37, 44},
    {35, 46, 57, 68},
  };

  auto d = Dot(a, b);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 4; j++) {
      EXPECT_EQ(d[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixDot_12_22)
{
  MatrixN<float> a = {
    {1, 2},
  };
  MatrixN<float> b = {
    {1, 2},
    {3, 4},
  };
  MatrixN<float> c = {
    {7, 10},
  };

  auto d = Dot(a, b);
  for (int i = 0; i < 1; i++)
  {
    for (int j = 0; j < 2; j++) {
      EXPECT_EQ(d[i][j], c[i][j]);
    }
  }
}

TEST(MatrixNTest, MatrixMultiplication_12_12)
{
  MatrixN<float> a = {
    {1, 2},
  };
  MatrixN<float> b = {
    {3, 4},
  };
  MatrixN<float> c = {
    {11},
  };

  auto d = Dot(a, b);
  EXPECT_EQ(d[0][0], c[0][0]);
}

TEST(MatrixNTest, DeeplearningZero)
{
  const auto X = MatrixN<float>({1.0, 0.5});
  const auto W1 = MatrixN<float>({
                                   {0.1, 0.3, 0.5},
                                   {0.2, 0.4, 0.6}
                                 });
  const auto B1 = MatrixN<float>({0.1, 0.2, 0.3});
  auto A1 = Dot(X, W1);
  A1 += B1;

  const auto ans = MatrixN<float>({0.3, 0.7, 1.1});

  for (int i = 0; i < 3; i++)
  {
    EXPECT_EQ(A1[0][i], ans[0][i]);
  }
}
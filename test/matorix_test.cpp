//
// Created by toru on 2023/12/26.
//
#include <gtest/gtest.h>
#include "nagatolib.hpp"
using namespace nagato;

TEST(MatrixTest, MatrixAddition)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {6, 8},
    {10, 12},
  };

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j] + b[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixSubtraction)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {-4, -4},
    {-4, -4},
  };

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j] - b[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixMultiplication)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {5, 12},
    {21, 32},
  };

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j] * b[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixDivision)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {0.2, 0.3333333333333333},
    {0.42857142857142855, 0.5},
  };

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j] / b[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixAdditionAssignment)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {6, 8},
    {10, 12},
  };

  a += b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixSubtractionAssignment)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {-4, -4},
    {-4, -4},
  };

  a -= b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixMultiplicationAssignment)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {5, 12},
    {21, 32},
  };

  a *= b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixDivisionAssignment)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };
  Matrix<float, 2, 2> c = {
    {0.2, 0.3333333333333333},
    {0.42857142857142855, 0.5},
  };

  a /= b;

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixDot)
{
  Matrix<float, 2, 2> a = {
    {1, 2},
    {3, 4},
  };
  Matrix<float, 2, 2> b = {
    {5, 6},
    {7, 8},
  };

  Matrix<float, 2, 2> c = {
    {19, 22},
    {43, 50},
  };

  const auto ans = Dot(a, b);

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    {
      EXPECT_EQ(ans[i][j], c[i][j]);
    }
  }
}

TEST(MatrixTest, MatrixDot_23_32)
{
  Matrix<float, 2, 3> a = {
    {1, 2, 3},
    {4, 5, 6},
  };
  Matrix<float, 3, 2> b = {
    {7, 8},
    {9, 10},
    {11, 12},
  };

  Matrix<float, 2, 2> c = {
    {58, 64},
    {139, 154},
  };

  const auto ans = Dot(a, b);
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++)
    { EXPECT_EQ(ans[i][j], c[i][j]); }
  }
}

TEST(MatrixTest, MatrixDot_32_23)
{
  Matrix<float, 3, 2> a = {
    {1, 2},
    {3, 4},
    {5, 6},
  };
  Matrix<float, 2, 3> b = {
    {7, 8, 9},
    {10, 11, 12},
  };

  Matrix<float, 3, 3> c = {
    {27, 30, 33},
    {61, 68, 75},
    {95, 106, 117},
  };

  const auto ans = Dot(a, b);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    { EXPECT_EQ(ans[i][j], c[i][j]); }
  }
}

TEST(MatrixTest, MatrixDot_32_34)
{
  Matrix<float, 2, 3> a = {
    {1, 2, 3},
    {4, 5, 6},
  };
  Matrix<float, 3, 4> b = {
    {7, 8, 9, 10},
    {11, 12, 13, 14},
    {15, 16, 17, 18},
  };

  Matrix<float, 2, 4> c = {
    {74, 80, 86, 92},
    {173, 188, 203, 218},
  };

  const auto ans = Dot(a, b);
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 4; j++)
    { EXPECT_EQ(ans[i][j], c[i][j]); }
  }
}

TEST(MatrixTest, MatrixDot_12_22)
{
  Matrix<float, 1, 2> a = {
    {1, 2},
  };
  Matrix<float, 2, 2> b = {
    {3, 4},
    {5, 6},
  };

  Matrix<float, 1, 2> c = {
    {13, 16},
  };

  const auto ans = Dot(a, b);

  for (int i = 0; i < 1; i++)
  {
    for (int j = 0; j < 2; j++)
    { EXPECT_EQ(ans[i][j], c[i][j]); }
  }
}

TEST(MatrixTest, MatrixVector_12_22)
{
  Vector<float, 2> a = {
    3, 4,
  };
  Matrix<float, 2, 2> b = {
    {1, 2},
    {3, 4},
  };

  Vector<float, 2> c = {
    15, 22,
  };

  const auto ans = Dot(a, b);

  for (int i = 0; i < 2; i++)
  {
    EXPECT_EQ(ans[i], c[i]);
  }
}

TEST(MatrixTest, MatrixVector_22_21)
{
  Vector<float, 2> a = {
    3, 4,
  };
  Matrix<float, 2, 2> b = {
    {1, 2},
    {3, 4},
  };

  Vector<float, 2> c = {
    11, 25,
  };

  const auto ans = Dot(b, a);

  for (int i = 0; i < 2; i++)
  {
    EXPECT_EQ(ans[i], c[i]);
  }
}

TEST(MatrixTest, MatrixMultiplication_12_12)
{
  Matrix<float, 1, 2> a = {
    {1, 2},
  };
  Matrix<float, 1, 2> b = {
    {3, 4},
  };

  Matrix<float, 1, 2> c = {
    {3, 8},
  };

  const auto ans = a * b;

  for (int i = 0; i < 1; i++)
  {
    for (int j = 0; j < 1; j++)
    { EXPECT_EQ(ans[i][j], c[i][j]); }
  }
}

TEST(MatrixTest, DeeplearningZero)
{
  using namespace nagato;
  const auto X = Vectorf<2>({1.0, 0.5});
  const auto W1 = Matrixf<2, 3>({
                                  {0.1, 0.3, 0.5},
                                  {0.2, 0.4, 0.6}
                                });
  const auto B1 = Vectorf<3>({0.1, 0.2, 0.3});
  auto A1 = Dot(X, W1);
  A1 += B1;

  const auto ans = Vectorf<3>({0.3, 0.7, 1.1});

  for (int i = 0; i < 3; i++)
  {
    EXPECT_EQ(A1[0][i], ans[0][i]);
  }

}
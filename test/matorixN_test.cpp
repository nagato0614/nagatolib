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





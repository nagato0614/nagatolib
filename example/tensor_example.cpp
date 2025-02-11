#include "nagatolib.hpp"

using namespace nagato;
void two_layer_network()
{
  Tensor X = Tensor::Ones({1, 2});
  X(0, 0) = 1.0;
  X(0, 1) = 0.5;

  Tensor W1 = Tensor::Ones({2, 3});
  W1(0, 0) = 0.1;
  W1(0, 1) = 0.3;
  W1(0, 2) = 0.5;
  W1(1, 0) = 0.2;
  W1(1, 1) = 0.4;
  W1(1, 2) = 0.6;

  Tensor B1 = Tensor::Ones({1, 3});
  B1(0, 0) = 0.1;
  B1(0, 1) = 0.2;
  B1(0, 2) = 0.3;

  Tensor::Print(X);
  Tensor::Print(W1);
  Tensor::Print(B1);

  Tensor A0 = Tensor::Matmul(X, W1);
  Tensor::Print(A0);
  Tensor A1 = A0 + B1;
  Tensor::Print(A1);

  Tensor Z1 = Tensor::Sigmoid(A1);
  Tensor::Print(Z1);

  Tensor W2 = Tensor::Ones({3, 2});
  W2(0, 0) = 0.1;
  W2(0, 1) = 0.4;
  W2(1, 0) = 0.2;
  W2(1, 1) = 0.5;
  W2(2, 0) = 0.3;
  W2(2, 1) = 0.6;

  Tensor B2 = Tensor::Ones({1, 2});
  B2(0, 0) = 0.1;
  B2(0, 1) = 0.2;

  Tensor::PrintShape(Z1);
  Tensor::PrintShape(W2);
  Tensor::PrintShape(B2);
  std::cout << std::endl;

  Tensor A2 = Tensor::Matmul(Z1, W2) + B2;
  Tensor::Print(A2);

  Tensor Z2 = Tensor::Sigmoid(A2);
  Tensor::Print(Z2);

  Tensor W3 = Tensor::Ones({2, 2});
  W3(0, 0) = 0.1;
  W3(0, 1) = 0.3;
  W3(1, 0) = 0.2;
  W3(1, 1) = 0.4;

  Tensor B3 = Tensor::Ones({1, 2});
  B3(0, 0) = 0.1;
  B3(0, 1) = 0.2;

  Tensor A3 = Tensor::Matmul(Z2, W3) + B3;
  Tensor::Print(A3);
}

Tensor MeanSquaredError(const Tensor &y, const Tensor &t)
{
  Tensor::IsSameShape(y, t);
  const Tensor diff = y - t;
  const Tensor square = diff * diff;
  const Tensor sum = Tensor::Sum(square) * 0.5;
  return sum;
}

// バッチ対応の交差エントロピー誤差
Tensor CrossEntropyError(const Tensor &y, const Tensor &t)
{
  constexpr float delta = 1e-7;
  Tensor::IsSameShape(y, t);
  const Tensor result = t * Tensor::Log(y + delta);
  return -Tensor::Sum(result);
}


int main()
{
  // ２乗誤差のテスト
  Tensor ans = Tensor::Zeros({2, 10});
  ans(0, 2) = 1.0;
  ans(1, 2) = 1.0;
  Tensor y = Tensor::Zeros({2, 10});
  // 0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0
  y(0, 0) = 0.1;
  y(0, 1) = 0.05;
  y(0, 2) = 0.6;
  y(0, 3) = 0.0;
  y(0, 4) = 0.05;
  y(0, 5) = 0.1;
  y(0, 6) = 0.0;
  y(0, 7) = 0.1;
  y(0, 8) = 0.0;
  y(0, 9) = 0.0;
  // 0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0
  y(1, 0) = 0.1;
  y(1, 1) = 0.05;
  y(1, 2) = 0.1;
  y(1, 3) = 0.0;
  y(1, 4) = 0.05;
  y(1, 5) = 0.1;
  y(1, 6) = 0.0;
  y(1, 7) = 0.6;
  y(1, 8) = 0.0;
  y(1, 9) = 0.0;

  Tensor::Print(ans);
  Tensor::Print(y);

  const Tensor result = MeanSquaredError(y, ans);
  Tensor::Print(result);

  // 交差エントロピー誤差のテスト
  const Tensor cross_result = CrossEntropyError(y, ans);
  Tensor::Print(cross_result);
}

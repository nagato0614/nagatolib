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

/**
 * @brief 乗算を取り扱うレイヤー
 */
class MulLayer
{
  public:
    MulLayer() = default;

    Tensor forward(const Tensor &x, const Tensor &y)
    {
      this->x = x;
      this->y = y;
      return x * y;
    }

    std::pair<Tensor, Tensor> backward(const Tensor &dout)
    {
      const Tensor dx = dout * y;
      const Tensor dy = dout * x;
      return std::make_pair(dx, dy);
    }

  private:
    Tensor x;
    Tensor y;
};

/**
 * @brief 加算を取り扱うレイヤー  
 */
class AddLayer
{
  public:
    AddLayer() = default;

    Tensor forward(const Tensor &x, const Tensor &y)
    {
      return x + y;
    }

    std::pair<Tensor, Tensor> backward(const Tensor &dout)
    {
      const Tensor dx = dout * 1.f;
      const Tensor dy = dout * 1.f;
      return std::make_pair(dx, dy);
    }
};

/**
 * @brief LeRU レイヤー
 */
class ReLU
{
  public:
    ReLU()
    {
      mask = [](const Tensor::value_type &x) { return x > 0 ? x : 0; };
    }

    Tensor forward(const Tensor &x)
    {
      return Tensor::Transform(x, mask);
    }

    Tensor backward(const Tensor &dout)
    {
      return Tensor::Transform(dout, mask);
    }

  private:
    std::function<Tensor::value_type(Tensor::value_type)> mask;
};

/**
 * @brief シグモイドレイヤー
 */
class Sigmoid
{
  public:
    Sigmoid() = default;

    Tensor forward(const Tensor &x)
    {
      out = 1.0 / (1.0 + Tensor::Exp(-x));
      return out;
    }

    Tensor backward(const Tensor &dout)
    {
      return dout * (1.0 - out) * out;
    }

  private:
    Tensor out;
};

int main()
{
  // 乗算レイヤーのテスト

  Tensor apple = Tensor::FromArray({100});
  Tensor apple_num = Tensor::FromArray({2});
  Tensor orange = Tensor::FromArray({150});
  Tensor orange_num = Tensor::FromArray({3});
  Tensor tax = Tensor::FromArray({1.1});

  MulLayer mul_apple_layer;
  MulLayer mul_orange_layer;
  MulLayer mul_tax_layer;
  AddLayer add_orange_layer;

  Tensor apple_price = mul_apple_layer.forward(apple, apple_num);
  Tensor orange_price = mul_orange_layer.forward(orange, orange_num);
  Tensor all_price = add_orange_layer.forward(apple_price, orange_price);
  Tensor price = mul_tax_layer.forward(all_price, tax);

  Tensor dprice = Tensor::FromArray({1});
  auto [dall_price, dtax] = mul_tax_layer.backward(dprice);
  auto [dapple_price, dorange_price] = add_orange_layer.backward(dall_price);
  auto [dapple, dapple_num] = mul_apple_layer.backward(dapple_price);
  auto [dorange, dorange_num] = mul_orange_layer.backward(dorange_price);

  Tensor::Print(price);
  Tensor::Print(dapple_num);
  Tensor::Print(dapple);
  Tensor::Print(dorange);
  Tensor::Print(dorange_num);
  Tensor::Print(dtax);
}

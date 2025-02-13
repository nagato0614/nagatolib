//
// Created by toru on 2025/02/12.
//

#include "network.hpp"
#include "tensor.hpp"
#include <algorithm>

namespace nagato
{

Tensor MeanSquaredError(const Tensor &y, const Tensor &t)
{
  Tensor::IsSameShape(y, t);
  const Tensor diff = y - t;
  const Tensor square = diff * diff;
  const Tensor sum = Tensor::Sum(square) * 0.5;
  return sum;
}

Tensor CrossEntropyError(const Tensor &y, const Tensor &t)
{
  // 入力テンソル y が (N, 1, D) の形状である場合, 1次元の軸を削除する.
  Tensor y_copy = y;
  if (y.shape().size() == 3 && y.shape()[1] == 1)
  {
    y_copy = y_copy.Reshape({y_copy.shape()[0], y_copy.shape()[2]});
  }
  Tensor::IsSameShape(y_copy, t);
  constexpr Tensor::value_type delta = 1e-7f;
  const auto &y_shape = y_copy.shape();
  size_t dim = y_shape.size();

  // 1次元の場合はバッチサイズ1として処理する
  if (dim == 1)
  {
    size_t num_classes = y_copy.storage().size();
    // 教師データが one-hot vector の場合、argmax で正解ラベルのインデックスを取得
    size_t label_index = 0;
    Tensor::value_type max_val = t(0);
    for (size_t i = 1; i < num_classes; ++i)
    {
      if (t(i) > max_val)
      {
        max_val = t(i);
        label_index = i;
      }
    }
    Tensor::value_type loss = -std::log(y_copy(label_index) + delta);
    return Tensor::FromArray({loss});
  }
  // 2次元の場合 (バッチサイズ, クラス数)
  else if (dim == 2)
  {
    size_t batch_size = y_shape[0];
    size_t num_classes = y_shape[1];

    // 教師データが one-hot vector の場合、全要素数が y と同じになる前提で処理する
    if (t.storage().size() == y.storage().size())
    {
      std::vector<size_t> label_indices(batch_size, 0);
      for (size_t i = 0; i < batch_size; ++i)
      {
        Tensor::value_type max_val = t(i, 0);
        size_t max_index = 0;
        // 各行（サンプル）ごとに正解ラベルのインデックスを判定
        for (size_t j = 1; j < num_classes; ++j)
        {
          if (t(i, j) > max_val)
          {
            max_val = t(i, j);
            max_index = j;
          }
        }
        label_indices[i] = max_index;
      }

      Tensor::value_type total_loss = 0.0f;
      for (size_t i = 0; i < batch_size; ++i)
      {
        total_loss += -std::log(y_copy(i, label_indices[i]) + delta);
      }
      total_loss /= static_cast<Tensor::value_type>(batch_size); // バッチ平均
      return Tensor::FromArray({total_loss});
    }
    else
    {
      // もし教師データがすでに正解クラスの確率情報（one-hot でない場合）なら、要素ごとの積を計算
      Tensor::value_type total_loss = 0.0f;
      const auto &y_storage = y_copy.storage();
      const auto &t_storage = t.storage();
      for (size_t i = 0; i < y_storage.size(); ++i)
      {
        total_loss += -std::log(y_storage[i] + delta) * t_storage[i];
      }
      total_loss /= static_cast<Tensor::value_type>(batch_size);
      return Tensor::FromArray({total_loss});
    }
  }
  else
  {
    throw std::invalid_argument("Tensor must be 1D or 2D for CrossEntropyError");
  }
}

Tensor Layer::get_dW() const
{
  return this->dW;
}

Tensor Layer::get_db() const
{
  return this->db;
}
ReLU::ReLU()
{
}

Tensor ReLU::forward(const Tensor &x)
{
  // backward で利用するため、入力の値（または mask）を保持
  this->input_ = x;
  Tensor result = Tensor::Zeros(x.shape());
  auto &x_storage = x.storage();
  auto &result_storage = result.storage();
  for (std::size_t i = 0; i < x_storage.size(); ++i) {
    result_storage[i] = std::max(static_cast<Tensor::value_type>(0), x_storage[i]);
  }
  return result;
}

Tensor ReLU::forward(const Tensor &x, const Tensor &y)
{
  throw std::invalid_argument("ReLU layer does not support two input tensors");
}

Tensor ReLU::backward(const Tensor &dout)
{
  // 保存しておいた入力に基づいて勾配を計算する:
  // x > 0 なら dout のまま、x <= 0 なら 0 とする
  Tensor dx = Tensor::Zeros(dout.shape());
  auto &dx_storage = dx.storage();
  auto &dout_storage = dout.storage();
  auto &input_storage = this->input_.storage();
  for (std::size_t i = 0; i < input_storage.size(); ++i) {
    dx_storage[i] = (input_storage[i] > 0) ? dout_storage[i] : 0.0f;
  }
  return dx;
}

Tensor ReLU::ReluFunction(const Tensor &x)
{
  return Tensor::Transform(x, [](const Tensor::value_type &x) { return x > 0 ? x : 0; });
}

Tensor Sigmoid::forward(const Tensor &x)
{
  out = 1.0 / (1.0 + Tensor::Exp(-x));
  return out;
}

Tensor Sigmoid::forward(const Tensor &x, const Tensor &y)
{
  throw std::invalid_argument("Sigmoid layer does not support two input tensors");
}

Tensor Sigmoid::backward(const Tensor &dout)
{
  return dout * (1.0 - out) * out;
}

Affine::Affine(const std::shared_ptr<Tensor> &W, const std::shared_ptr<Tensor> &b): W(W), b(b)
{
}


Tensor Affine::forward(const Tensor &x)
{
  // N : バッチサイズ
  // D : 入力次元数
  // H : 隠れ層の次元数
  // 入力 x は shape (N, D) で与えられるため (N, 1, D) に reshape する
  // Wは (D, H) で与えられるため、(N, D, H) に reshape する
  // すでに (N, 1, D) の形状なら reshape しなくてもよい
  if (x.shape().size() == 3 && x.shape()[1] == 1)
  {
    this->x = x;
  }
  else
  {
    Tensor x_copy = x;
    this->x = x_copy.Reshape({x_copy.shape()[0], 1, x_copy.shape()[1]});
  }

  Tensor broadcast_W = Tensor::Tile(*W, x.shape()[0]);
  Tensor z = Tensor::Matmul(this->x, broadcast_W);
  // バイアス b は shape (1, output) で与えられているので、バッチサイズに合わせて展開する
  Tensor broadcast_b = Tensor::Tile(*b, x.shape()[0]);
  return z + broadcast_b;
}

Tensor Affine::forward(const Tensor &x, const Tensor &y)
{
  throw std::invalid_argument("Affine layer does not support two input tensors");
}

Tensor Affine::backward(const Tensor &dout)
{
  Tensor W_T = Tensor::Transpose(*W);
  Tensor x_T = Tensor::Transpose(x);

  // W_T は (D, H) なので、(N, D, H) に ブロードキャストする
  if (dout.shape().size() == 3)
  {
    W_T = Tensor::Tile(W_T, dout.shape()[0]);
  }

  Tensor dx = Tensor::Matmul(dout, W_T);
  this->dW = Tensor::Matmul(x_T, dout);
  this->db = Tensor::Sum(dout, 0);
  return dx;
}
Tensor SoftmaxWithLoss::forward(const Tensor &x)
{
  throw std::invalid_argument("SoftmaxWithLoss layer does not support one input tensor");
}
Tensor SoftmaxWithLoss::forward(const Tensor &x, const Tensor &t)
{
  this->t = t;
  this->y = Tensor::Softmax(x);
  this->loss = CrossEntropyError(y, t);
  return this->loss;
}

Tensor SoftmaxWithLoss::backward(const Tensor &dout)
{
  Tensor dx = (y - t);
  Tensor::value_type batch_size = static_cast<Tensor::value_type>(t.shape()[0]);
  Tensor result = dx / batch_size;
  return result;
}

TwoLayerNet::TwoLayerNet(const std::size_t input_size,
  const std::size_t hidden_size,
  const std::size_t output_size,
  const Tensor::value_type weight_init_std)
{
  this->params = std::vector<std::pair<std::string, std::shared_ptr<Tensor> > >();
  this->layers = std::vector<std::pair<std::string, std::unique_ptr<Layer> > >();

  // 重みの初期化
  this->params.emplace_back("W1",
                            std::make_shared<Tensor>(
                              Tensor::RandomNormal({
                                input_size, hidden_size
                              }) * weight_init_std));
  this->params.emplace_back("b1",
                            std::make_shared<Tensor>(
                              Tensor::RandomNormal({1, hidden_size}) * weight_init_std));
  this->params.emplace_back("W2",
                            std::make_shared<Tensor>(
                              Tensor::RandomNormal({hidden_size, output_size}) *
                              weight_init_std));
  this->params.emplace_back("b2",
                            std::make_shared<Tensor>(
                              Tensor::RandomNormal({1, output_size}) * weight_init_std));

  // レイヤーの生成
  this->layers.emplace_back("Affine1",
                            std::make_unique<Affine>(this->params[0].second,
                                                     this->params[1].second));
  this->layers.emplace_back("ReLU", std::make_unique<ReLU>());
  this->layers.emplace_back("Affine2",
                            std::make_unique<Affine>(this->params[2].second,
                                                     this->params[3].second));
}
Tensor TwoLayerNet::predict(const Tensor &x)
{
  Tensor result = x; // ローカル変数にコピー
  for (auto &layer : this->layers)
  {
    result = layer.second->forward(result);
    if (Tensor::IsNan(result))
    {
      std::cout << "nan" << std::endl;
    }
  }
  return result;
}

Tensor TwoLayerNet::loss(const Tensor &x, const Tensor &t)
{
  Tensor y = this->predict(x);
  return this->last_layer.forward(y, t);
}

Tensor::value_type TwoLayerNet::loss_batch(const Tensor &x, const Tensor &t)
{
  Tensor y = this->predict(x);
  auto fow = this->last_layer.forward(y, t);
  auto avg_loss = Tensor::Mean(fow);
  return avg_loss(0);
}
Tensor::value_type TwoLayerNet::accuracy(const Tensor &x, const Tensor &t)
{
  Tensor y = this->predict(x);
  Tensor result = y.Argmax();
  Tensor ans = t.Argmax();
  Tensor equal = result == ans;
  std::cout << "correct / total: " << Tensor::Sum(equal)(0) << " / " << x.shape()[0] <<
    std::endl;
  return Tensor::Sum(equal)(0) / x.shape()[0];
}

std::vector<std::pair<std::string, Tensor>> TwoLayerNet::numerical_gradient(
  const Tensor &x,
  const Tensor &t)
{
  std::vector<std::pair<std::string, Tensor>> grads;
  // 各パラメータごとに数値微分を行う
  for (std::size_t i = 0; i < this->params.size(); i++) {
    auto &param = this->params[i];
    // 現在のパラメータのコピーを取得（元に戻すため）
    Tensor original = *(param.second);

    auto f = [this, &x, &t, original, i](const Tensor &param_var) -> Tensor {
      // 該当パラメータを候補値に置き換える
      *(this->params[i].second) = param_var;
      Tensor loss_tensor = this->loss(x, t);
      // 値を元に戻す
      *(this->params[i].second) = original;
      // loss_tensor は1要素のテンソルであると仮定
      return Tensor::FromArray({loss_tensor(0)});
    };

    grads.emplace_back(param.first, numerical_gradient_(f, *param.second));
  }
  return grads;
}

std::vector<std::pair<std::string, Tensor>> TwoLayerNet::gradient(const Tensor &x,
  const Tensor &t)
{
  // forward
  this->loss(x, t);

  // backward
  constexpr Tensor::value_type dout = 1;
  Tensor dout_tensor = Tensor::FromArray({dout});
  Tensor dx = this->last_layer.backward(dout_tensor);

  // 逆順でレイヤーを処理する
  for (int i = this->layers.size() - 1; i >= 0; --i)
  {
    dx = this->layers[i].second->backward(dx);
  }

  // 設定
  std::vector<std::pair<std::string, Tensor> > grads;
  grads.emplace_back("W1", this->layers[0].second->get_dW());
  grads.emplace_back("b1", this->layers[0].second->get_db());
  grads.emplace_back("W2", this->layers[2].second->get_dW());
  grads.emplace_back("b2", this->layers[2].second->get_db());
  return grads;
}

} // namespace nagato
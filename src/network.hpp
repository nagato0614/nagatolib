//
// Created by toru on 2025/02/12.
//

#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "nagatolib.hpp"

namespace nagato
{
Tensor MeanSquaredError(const Tensor &y, const Tensor &t);

// バッチ対応の交差エントロピー誤差
Tensor CrossEntropyError(const Tensor &y, const Tensor &t);

/**
 * @brief レイヤーの基底クラス
 */
class Layer
{
  public:
    virtual Tensor forward(const Tensor &x) = 0;
    virtual Tensor forward(const Tensor &x, const Tensor &y) = 0;
    virtual Tensor backward(const Tensor &dout) = 0;
    virtual ~Layer() = default;
    Tensor get_dW() const;

    Tensor get_db() const;

  protected:
    Tensor dW;
    Tensor db;
};

/**
 * @brief LeRU レイヤー
 */
class ReLU : public Layer
{
  public:
    ReLU();

    Tensor forward(const Tensor &x) override;

    Tensor forward(const Tensor &x, const Tensor &y) override;

    Tensor backward(const Tensor &dout) override;

  private:
    Tensor ReluFunction(const Tensor &x);
    Tensor input_;
};

/**
 * @brief シグモイドレイヤー
 */
class Sigmoid : public Layer
{
  public:
    Sigmoid() = default;

    Tensor forward(const Tensor &x) override;

    Tensor forward(const Tensor &x, const Tensor &y) override;

    Tensor backward(const Tensor &dout) override;

  private:
    Tensor out;
};

/**
 * @brief Affine レイヤー
 */
class Affine : public Layer
{
  public:
    Affine(const std::shared_ptr<Tensor> &W, const std::shared_ptr<Tensor> &b);

    Tensor forward(const Tensor &x) override;

    Tensor forward(const Tensor &x, const Tensor &y) override;

    Tensor backward(const Tensor &dout) override;

  private:
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;
    Tensor x;
};

/**
 * @brief Softmax-with-Loss レイヤー
 */
class SoftmaxWithLoss : public Layer
{
  public:
    SoftmaxWithLoss() = default;

    Tensor forward(const Tensor &x) override;

    Tensor forward(const Tensor &x, const Tensor &t) override;

    Tensor backward(const Tensor &dout) override;

  private:
    Tensor y;
    Tensor t;
    Tensor loss;
};

/**
 * @brief 重みパラメータに対する勾配を計算する関数.
 * @note バッチ処理に対応している
 * @param func 勾配を計算する関数
 * @param x 入力. 一番最初の次元はバッチサイズ
 * @return 勾配
 */
inline Tensor numerical_gradient_(std::function<Tensor(const Tensor &)> func, const Tensor &x)
{
  constexpr float h = 1e-3;

  // x と同じ形状を持つゼロ初期化のテンソルを作成する
  Tensor grad = Tensor::Zeros(x.shape());
  // x の変更可能なコピーを作成する
  Tensor x_copy = x;

  // Tensor のストレージ全体（全要素）でループ
  for (std::size_t idx = 0; idx < x_copy.storage().size(); ++idx)
  {
    // 残りの処理データを10つごとに表示
    if (idx % 10 == 0)
    {
      std::cout << "\r";
      std::cout << idx << " / " << x_copy.storage().size();
      std::cout.flush();
    }

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
  std::cout << std::endl;

  return grad;
}

/**
 * @brief 2層ニューラルネットワーク
 */
class TwoLayerNet
{
  public:
    TwoLayerNet(
      const std::size_t input_size,
      const std::size_t hidden_size,
      const std::size_t output_size,
      const float weight_init_std = 0.01
    );

    Tensor predict(const Tensor &x);

    Tensor loss(const Tensor &x, const Tensor &t);

    /**
     * @brief バッチで平均したloss を計算する
     * @param x 入力
     * @param t 正解ラベル
     * @return 平均loss
     */
    float loss_batch(const Tensor &x, const Tensor &t);

    float accuracy(const Tensor &x, const Tensor &t);

    std::vector<std::pair<std::string, Tensor> > numerical_gradient(
      const Tensor &x,
      const Tensor &t);

    std::vector<std::pair<std::string, Tensor> > gradient(const Tensor &x, const Tensor &t);

    std::vector<std::pair<std::string, std::shared_ptr<Tensor> > > params;
    std::vector<std::pair<std::string, std::unique_ptr<Layer> > > layers;
    SoftmaxWithLoss last_layer;
};

inline Tensor OneHot(const Tensor &x, const std::size_t &num_classes)
{
  Tensor one_hot = Tensor::Zeros({x.shape()[0], num_classes});
  for (std::size_t i = 0; i < x.shape()[0]; ++i)
  {
    one_hot(i, x(i, 0)) = 1.0;
  }
  return one_hot;
}

/**
 * @brief MNIST のデータを表示する.
 * @note データは (28 x 28) の行列で表示する
 * @param x データ (28 x 28)
 * @param label ラベル
 */
inline void PrintMNIST(const Tensor &x, const Tensor &label)
{
  Tensor x_copy = x;

  // x が (28, 28) の行列であることを確認し違う場合は変形する
  if (x.shape()[0] != 28 || x.shape()[1] != 28)
  {
    x_copy = x_copy.Reshape({28, 28});
  }

  // label が (1, 1) の行列の場合one-hot に変換する
  Tensor label_copy = label;
  if (label.shape()[0] == 1 && label.shape()[1] == 1)
  {
    label_copy = OneHot(label, 10);
  }

  // 128 以上のデータは@, それ以外は. で表示する
  for (std::size_t i = 0; i < 28; ++i)
  {
    for (std::size_t j = 0; j < 28; ++j)
    {
      if (x_copy(i, j) >= 128)
      {
        std::cout << "@ ";
      }
      else
      {
        std::cout << ". ";
      }
    }
    std::cout << std::endl;
  }
  Tensor::Print(label_copy);
}
} // namespace nagato

#endif //NETWORK_HPP

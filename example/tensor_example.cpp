#include "nagatolib.hpp"
#include <memory>
#include "network.hpp"

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

int main()
{
  // Mnist のデータセットをcsvから読み込む
  Tensor train_data = Tensor::FromCSV("../train_data.csv");
  Tensor train_label = Tensor::FromCSV("../train_label.csv");
  Tensor test_data = Tensor::FromCSV("../test_data.csv");
  Tensor test_label = Tensor::FromCSV("../test_label.csv");
  Tensor::PrintShape(train_data);
  Tensor::PrintShape(train_label);
  Tensor::PrintShape(test_data);
  Tensor::PrintShape(test_label);

  // 学習用データを 0 ~ 1.0 に正規化する
  train_data = train_data / 255.0;
  test_data = test_data / 255.0;

  Tensor train_label_one_hot = OneHot(train_label, 10);
  Tensor test_label_one_hot = OneHot(test_label, 10);

  // データを一つ表示する
  PrintMNIST(train_data.Slice(0), train_label.Slice(0));

  // ニューラルネットワークの生成
  TwoLayerNet net(784, 50, 10);

  constexpr std::size_t iter_num = 100000;
  constexpr std::size_t batch_size = 100;
  constexpr Tensor::value_type learning_rate = 0.1;

  // 学習データの総数を取得（例としてtrain_dataの最初の次元がサンプル数だとする）
  std::size_t data_size = train_data.shape()[0];

  // 全サンプルのインデックスベクトルを作成
  std::vector<std::size_t> all_indexes(data_size);
  std::iota(all_indexes.begin(), all_indexes.end(), 0);

  // 学習開始前に全体をシャッフルしておく
  std::mt19937 rng(std::random_device{}());
  std::shuffle(all_indexes.begin(), all_indexes.end(), rng);

  for (std::size_t i = 0; i < iter_num; ++i)
  {
    // 必要ならエポック毎に再度全体をシャッフルする
    if (i % (data_size / batch_size) == 0) {
        std::shuffle(all_indexes.begin(), all_indexes.end(), rng);
    }

    // ミニバッチの開始インデックスを計算（例: 連続してとる場合）
    std::size_t start = (i * batch_size) % data_size;
    std::vector<Tensor> x_batches;
    std::vector<Tensor> t_batches;

    for (std::size_t j = 0; j < batch_size; ++j)
    {
      std::size_t index = all_indexes[start + j];
      Tensor x_slice = train_data.Slice(index);
      Tensor t_slice = train_label_one_hot.Slice(index);

      x_batches.emplace_back(x_slice);
      t_batches.emplace_back(t_slice);
    }

    Tensor x_batch = Tensor::Concat(x_batches);
    Tensor t_batch = Tensor::Concat(t_batches);

    std::vector<std::pair<std::string, Tensor> > grads = net.gradient(x_batch, t_batch);

    // パラメータの更新
    for (std::size_t j = 0; j < grads.size(); ++j)
    {
      auto &param = net.params[j].second;
      auto &grad = grads[j].second;

      *param = *param - grad * learning_rate;
    }

    if (i % 100 == 0)
    {
      std::cout << " --- iter: " << i << " ---" << std::endl;
      std::cout << "loss: " << net.loss_batch(x_batch, t_batch) << std::endl;
      std::cout << "train accuracy: ";
      net.accuracy(x_batch, t_batch);

      // test_accuracy
      if (i % 1000 == 0)
      {
        std::cout << "test_accuracy: ";
        const auto acc = net.accuracy(test_data, test_label_one_hot);
        std::cout << "test_accuracy: " << acc << std::endl;

        // テストデータの loss を計算
        const auto test_loss = net.loss_batch(test_data, test_label_one_hot);
        std::cout << "test_loss: " << test_loss << std::endl;
      }
    }

    if ((i + 1) % 10 == 0)
    {
      // break;
    }
  }
}

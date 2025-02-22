//
// Created by toru on 2025/02/09.
//

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cassert>
#include <functional>
#include <optional>
#include <vector>
#include <algorithm>
#include <iostream>

#define NAGATO_OPENMP



namespace nagato{

/**
 * numpy の ndarray に相当するクラス
 */
class Tensor {
public:
 using value_type = double;
 using shape_type = std::vector<std::size_t>;
 using strides_type = std::vector<std::size_t>;
 using storage_type = std::vector<value_type>;

  Tensor();

  Tensor(const shape_type &shape);

  /**
   * @brief テンソルの形状を取得する
   * @return 形状
   */
  const shape_type &shape() const;

  /**
   * @brief テンソルのストライドを取得する
   * @return ストライド
   */
  const strides_type &strides() const;

  /**
   * @brief テンソルのデータを取得する
   * @return データ
   */
  const storage_type &storage() const;

  /**
   * @brief テンソルのデータを取得する (非 const 版)
   * @return データ
   */
  storage_type &storage();

  /**
   * @brief テンソルへの要素アクセス (const 版)
   * @param indices インデックス
   * @return 要素
   */
  template <typename... Indices>
  const value_type &operator()(Indices... indices) const {
    // 与えられたidndeces が shape_ と同じサイズであるかチェック
    if (sizeof...(Indices) != shape_.size()) {
      throw std::invalid_argument("index size must be equal to shape size");
    }

    return storage_[index_of({static_cast<std::size_t>(indices)...})];
  }

  /**
   * @brief テンソルへの要素アクセス (非 const 版)
   * @param indices インデックス
   * @return 要素
   */
  template <typename... Indices>
  value_type &operator()(Indices... indices) {
    return const_cast<value_type &>(const_cast<const Tensor &>(*this)(indices...));
  }
  
  /**
   * @brief テンソルの形状を変更する. 要素数は変更しない
   * @param new_shape 新しい形状
   * @return 新しいテンソル
   */
  Tensor Reshape(const shape_type &new_shape) const;

  /**
   * @brief テンソルの指定した次元を取り出す.
   * @param axis 取り出す次元
   * @return 取り出したテンソル
   */
  Tensor Slice(const std::size_t &axis) const;

  /**
   * @brief 指定した範囲のデータを取り出す
   * @param start 開始インデックス
   * @param end 終了インデックス. 終了インデックスのデータを含む
   * @return 取り出したテンソル
   */
  Tensor Slice(const std::size_t &start, const std::size_t &end) const;

  /**
   * @brief テンソルの最大値のインデックスを返す
   * @return 最大値のインデックス
   */
  Tensor Argmax() const;

  /**
   * @brief ゼロテンソルを作成する
   * @param shape 形状
   * @return ゼロテンソル
   */
  static Tensor Zeros(const shape_type &shape);

  /**
   * @brief テンソルを埋める
   * @param shape 形状
   * @param value 値
   * @return 埋めたテンソル
   */
  static Tensor Fill(const shape_type &shape, const value_type &value);

  /**
   * @brief 1のテンソルを作成する
   * @param shape 形状
   * @return 1のテンソル
   */
  static Tensor Ones(const shape_type &shape);

  /**
   * @brief 単位行列を作成する
   * @param shape 形状
   * @return 単位行列
   */
  static Tensor Eye(const shape_type &shape);

  /**
   * @brief テンソルの内積を計算する
   * @note テンソルはベクトルとして扱う, 2次元テンソルの場合は行ごとに内積を計算する. ブロードキャストは行わない
   * @param a テンソル
   * @param b テンソル
   * @return 内積
   */
  static Tensor Dot(const Tensor &a, const Tensor &b);

  /**
   * @brief 行列の積を計算する. 行列とベクトルの積も計算する.
   * @note ブロードキャストは行わない. 2次元の場合は行列同士の積, 3次元はバッチ行列同士の積を計算する. 
   * @param a テンソル
   * @param b テンソル
   * @return 行列の積
   */
  static Tensor Matmul(const Tensor &a, const Tensor &b);

  /**
   * @brief テンソルの総和を求める. shape_ の最後の要素を軸として総和を求める. 4次元以上のテンソルはサポートしない
   * @param a テンソル
   * @return 総和
   */
  static Tensor Sum(const Tensor &a);

  /**
   * @brief テンソルの総和を求める. shape_ の指定した軸を軸として総和を求める. 4次元以上のテンソルはサポートしない
   * @param a テンソル
   * @param axis 軸
   * @return 総和
   */
  static Tensor Sum(const Tensor &a, const std::size_t &axis);

  /**
   * @brief シグモイド関数を計算する
   * @param a テンソル
   * @return シグモイド関数
   */
  static Tensor Sigmoid(const Tensor &a);

  /**
   * @brief ReLU関数を計算する
   * @param a テンソル
   * @return ReLU関数
   */
  static Tensor ReLU(const Tensor &a);

  /**
   * @brief 指数関数を計算する
   * @param a テンソル
   * @return 指数関数
   */
  static Tensor Exp(const Tensor &a);

  /**
   * @brief 対数関数を計算する. 0 に近い値は 1e-7 に修正して計算する
   * @param a テンソル
   * @return 対数関数
   */
  static Tensor Log(const Tensor &a);

  /**
   * @brief Softmax関数を計算する
   * @note テンソルはベクトルとして扱う, 2次元テンソルの場合は行ごとにSoftmaxを計算する. ブロードキャストは行わない. 3次元以上のテンソルはサポートしない
   * @param a テンソル
   * @return Softmax関数
   */
  static Tensor Softmax(const Tensor &a);

  /**
   * @brief テンソルを表示する
   * @param a テンソル
   */
  static void Print(const Tensor &a);

  /**
   * @brief テンソルの形状を表示する
   * @param a テンソル
   */
  static void PrintShape(const Tensor &a);

  /**
   * @brief ２つのテンソルの形状が等しいかどうかをチェックする
   * @note 形状が等しくない場合, 例外を送出する
   * @param a テンソル
   * @param b テンソル
   */
  static void IsSameShape(const Tensor &a, const Tensor &b);

  /**
   * @brief ブロードキャスト可能かどうかをチェックする
   * @param a テンソル
   * @param b テンソル
   * @return ブロードキャスト可能な軸を返す. ブロードキャスト不可能な場合は-1を返す
   */
  static int IsBroadcastable(const Tensor &a, const Tensor &b);

  /**
   * @brief 乱数を生成する
   * @param shape 形状
   * @return 乱数
   */
  static Tensor Random(const shape_type &shape);

  /**
   * @brief 正規分布に従って乱数を生成する
   * @param shape 形状
   * @return 乱数
   */
  static Tensor RandomNormal(const shape_type &shape);

  /**
   * @brief 配列からテンソルを作成する
   * @note 配列のサイズが1の場合, 1次元テンソルとして作成する
   * @param array 配列
   * @return テンソル
   */
  static Tensor FromArray(const std::vector<value_type> &array);

  /**
   * @brief 配列からテンソルを作成する. 2次元テンソルとして作成する
   * @param array 配列
   * @return テンソル
   */
  static Tensor FromArray(const std::vector<std::vector<value_type>> &array);

  /**
   * @brief 配列からテンソルを作成する. 3次元テンソルとして作成する
   * @param array 配列
   * @return テンソル
   */
  static Tensor FromArray(const std::vector<std::vector<std::vector<value_type>>> &array);

  /**
   * @brief テンソルの要素ごとに関数を適用する
   * @param a テンソル
   * @param func 関数
   * @return 適用したテンソル
   */
  static Tensor Transform(const Tensor &a, const std::function<value_type(value_type)> &func);

  /**
   * @brief テンソルの転置を行う.
   * 次元数によって転置の方法が異なる
   *  - 1次元テンソル : 転置せず例外を送出する
   *  - 2次元テンソル : 転置する
   *  - 3次元テンソル : バッチ軸はそのままで, それ以外の軸を転置する
   *  - 4次元テンソル以降 : 転置せず例外を送出する
   * @param a テンソル
   * @return 転置したテンソル
   */
  static Tensor Transpose(const Tensor &a);

  /**
   * @brief 単項演算子のオーバーロード
   * @note テンソルの要素ごとに演算を行う
   */
  Tensor operator-() const;

  /**
   * @brief ２つのテンソルの値が等しいことを確認する
   * @param a テンソル
   * @param b テンソル
   */
  static bool Equal(const Tensor &a, const Tensor &b);

  /**
   * @brief テンソルをCSVファイルから読み込む
   * @param filename ファイル名
   * @return テンソル
   */
  static Tensor FromCSV(const std::string &filename);

  /**
   * @brief テンソルの平均を求める
   * @param a テンソル
   * @return 平均
   */
  static Tensor Mean(const Tensor &a);

  /**
   * @brief テンソルの絶対値を求める
   * @param a テンソル
   * @return 絶対値
   */
  static Tensor Abs(const Tensor &a);

  /**
   * @brief 複数のテンソルを組み合わせて一つのテンソルにする (単一引数の場合はそのまま返す)
   * @param first テンソル
   * @return 組み合わせたテンソル
   */
  static Tensor Concat(const Tensor &first)
  {
    return first;
  }

  /**
   * @brief 複数のテンソルを組み合わせて一つのテンソルにする
   * @param first 最初のテンソル
   * @param rest 残りのテンソル群
   * @return 組み合わせたテンソル
   */
  template <typename... Tensors>
  static Tensor Concat(const Tensor &first, const Tensors &...rest)
  {
    // すべてのテンソルの形状が同一であることをチェック
    ((void)((rest.shape() == first.shape())
            ? 0
            : throw std::invalid_argument("all tensors must have the same shape")), ...);
    
    // 引数の数をテンソルの個数として登録し, その後に元の形状を登録する
    shape_type new_shape = {1 + sizeof...(Tensors)};
    new_shape.insert(new_shape.end(), first.shape().begin(), first.shape().end());

    // 結果のテンソルを生成 (new_shape に基づいてストレージサイズも確保される)
    Tensor result(new_shape);
    std::size_t offset = 0;
    
    // 最初のテンソルのストレージをコピー
    std::copy(first.storage().begin(), first.storage().end(),
              result.storage().begin() + offset);
    offset += first.storage().size();
    
    // 残りのテンソル群をコピー (fold expression により展開)
    ((std::copy(rest.storage().begin(), rest.storage().end(),
                result.storage().begin() + offset),
      offset += rest.storage().size()), ...);
    
    return result;
  }

  /**
   * @brief 複数のテンソルを組み合わせて一つのテンソルにする
   * @param tensors テンソル群
   * @return 組み合わせたテンソル
   */
  static Tensor Concat(const std::vector<Tensor> &tensors);

  /**
   * @brief テンソルの最大値を求める
   * @param a テンソル
   * @return 最大値
   */
  static Tensor::value_type Max(const Tensor &a);

  /**
   * @brief テンソルの最小値を求める
   * @param a テンソル
   * @return 最小値
   */
  static Tensor::value_type Min(const Tensor &a);

  /**
   * @brief テンソルに nan が含まれているかどうかをチェックする
   * @param a テンソル
   * @return nan が含まれているかどうか
   */
  static bool IsNan(const Tensor &a);
  
  /**
   * @brief テンソルを繰り返し展開する
   * @param a テンソル
   * @param batch_size 繰り返し回数
   * @return 展開したテンソル
   */
  static Tensor Tile(const Tensor &a, std::size_t batch_size);
private:

  /**
   * @brief 現在設定されている形状をチェック
   * @return 形状が設定されているかどうか. 正しい場合は真となる
   * @note 形状が設定されていない場合, 形状を設定する必要がある
   */
  bool has_shape() const;

  /**
   * @brief 形状を設定してテンソルを初期化する. 同時にデータ領域も確保する
   * @param shape 形状
   */
  void set_shape(const shape_type &shape);

  /**
   * @brief インデックスをストライドに応じて乗算して加算していく
   * @param indices インデックス
   * @return インデックス
   */
  std::size_t index_of(const std::initializer_list<std::size_t> &indices) const {
    std::size_t index = 0;
    std::size_t i = 0;

    // インデックスをストライドに応じて乗算して加算していく
    for (std::size_t idx : indices) {
      // インデックスが shape_ の要素数と一致しない場合, エラーとする
      if (idx >= shape_[i]) {
        throw std::out_of_range("index is out of range");
      }

      index += idx * strides_[i++];
    }
    return index;
  }

  /// テンソルの形状
  shape_type shape_;

  /// テンソルのストライド
  strides_type strides_;

  /// テンソルのデータ
  storage_type storage_;
};

/**
 * @brief ブロードキャスト可能な二項演算を適用する
 * @param a テンソル
 * @param b テンソル
 * @param op 二項演算
 * @return 適用したテンソル
 */
template <typename BinaryOp>
Tensor ApplyBroadcastBinaryOp(const Tensor &a, const Tensor &b, BinaryOp op);

/**
 * @brief テンソルの加算
 * @param a テンソル
 * @param b テンソル
 * @return 加算結果
 */
Tensor operator+(const Tensor &a, const Tensor &b);

/**
 * @brief テンソルの加算
 * @param a テンソル
 * @param b スカラー
 * @return 加算結果
 */
Tensor operator+(const Tensor &a, const Tensor::value_type &b);
/**
 * @brief テンソルの加算
 * @param a テンソル
 * @param b スカラー
 * @return 加算結果
 */
Tensor operator+(const Tensor::value_type &a, const Tensor &b);

/**
 * @brief テンソルの減算
 * @param a テンソル
 * @param b テンソル
 * @return 減算結果
 */
Tensor operator-(const Tensor &a, const Tensor &b);

/**
 * @brief テンソルの減算
 * @param a テンソル
 * @param b スカラー
 * @return 減算結果
 */
Tensor operator-(const Tensor &a, const Tensor::value_type &b);

/**
 * @brief テンソルの減算
 * @param a テンソル
 * @param b スカラー
 * @return 減算結果
 */
Tensor operator-(const Tensor::value_type &a, const Tensor &b);
/**
 * @brief テンソルの乗算
 * @param a テンソル
 * @param b テンソル
 * @return 乗算結果
 */
Tensor operator*(const Tensor &a, const Tensor &b);

/**
 * @brief テンソルの乗算
 * @param a テンソル
 * @param b スカラー
 * @return 乗算結果
 */
Tensor operator*(const Tensor &a, const Tensor::value_type &b);

/**
 * @brief テンソルの乗算
 * @param a テンソル
 * @param b スカラー
 * @return 乗算結果
 */
Tensor operator*(const Tensor::value_type &a, const Tensor &b);

/**
 * @brief テンソルの除算
 * @note 0 で除算した場合, 0 になる
 * @param a テンソル
 * @param b テンソル
 * @return 除算結果
 */
Tensor operator/(const Tensor &a, const Tensor &b);

/**
 * @brief テンソルの除算
 * @param a テンソル
 * @param b スカラー
 * @return 除算結果
 */
Tensor operator/(const Tensor &a, const Tensor::value_type &b);  

/**
 * @brief テンソルの除算
 * @param a テンソル
 * @param b スカラー
 * @return 除算結果
 */
Tensor operator/(const Tensor::value_type &a, const Tensor &b);


/**
 * @brief テンソルの等号演算子. 等しい場合は1.0, 等しくない場合は0.0を返す
 * @param a テンソル
 * @param b テンソル
 * @return 等号演算結果
 */
Tensor operator==(const Tensor &a, const Tensor &b);


}  // namespace nagato

#endif // TENSOR_HPP

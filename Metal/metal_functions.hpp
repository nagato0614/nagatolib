//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_
#define NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_
#include "metal_base.hpp"
#include "metal_common.hpp"

namespace nagato::mtl
{
constexpr size_t DefaultThreadPerGroup = 256;

class MLASingleton
{
  public:
    static auto &GetInstance()
    {
      static MLASingleton instance;
      return instance;
    }

    auto &GetMetalBase();

  private:
    MLASingleton();
    ~MLASingleton() = default;

    std::unique_ptr<MetalBase> metal_base_;
};

/**
 * @brief MetalAdderFunction
 *
 * このクラスは要素ごとの加算をGPUで行うためのクラスです。
 * 最適化されていないためこのクラスの利用は非推奨です.
 */
class MetalAdderFunction
{
  public:
    explicit MetalAdderFunction(std::size_t length);
    ~MetalAdderFunction() = default;

    void operator()(
      const nFloat *inA,
      const nFloat *inB,
      nFloat *result
    );

  private:
    std::size_t buffer_length_;
    std::unique_ptr<MetalFunctionBase> add_arrays_;
};

class MetalSubFunction
{
  public:
    explicit MetalSubFunction(std::size_t length);
    ~MetalSubFunction() = default;

    void operator()(
      const nFloat *inA,
      const nFloat *inB,
      nFloat *result
    );

  private:
    std::size_t buffer_length_;
    std::unique_ptr<MetalFunctionBase> sub_arrays_;
};

class MetalMulFunction
{
  public:
    explicit MetalMulFunction(std::size_t length);
    ~MetalMulFunction() = default;

    void operator()(
      const nFloat *inA,
      const nFloat *inB,
      nFloat *result
    );

  private:
    std::size_t buffer_length_;
    std::unique_ptr<MetalFunctionBase> mul_arrays_;
};

class MetalDivFunction
{
  public:
    explicit MetalDivFunction(std::size_t length);
    ~MetalDivFunction() = default;

    void operator()(
      const nFloat *inA,
      const nFloat *inB,
      nFloat *result
    );

  private:
    std::size_t buffer_length_;
    std::unique_ptr<MetalFunctionBase> div_arrays_;
};

/**
 * @brief MetalAdderFunction
 *
 * このクラスは要素ごとに, 平方根を求めるくらすです
 * 最適化されていないためこのクラスの利用は非推奨です.
 */
class MetalSqrtFunction
{
  public:
    explicit MetalSqrtFunction(std::size_t length);
    ~MetalSqrtFunction() = default;

    void operator()(
      const nFloat *inA,
      nFloat *result
    );

  private:
    std::size_t buffer_length_;
    std::unique_ptr<MetalFunctionBase> sqrt_arrays_;
};

class MetalSumFunction
{
  public:
    explicit MetalSumFunction(std::size_t length);
    ~MetalSumFunction() = default;

    void operator()(
      const nFloat *inA,
      nFloat *result
    );

  private:
    std::size_t buffer_length_;
    std::unique_ptr<MetalFunctionBase> sum_arrays_;
};

class MetalSoftmaxFunction
{
  public:
    MetalSoftmaxFunction(std::size_t arrayLength);
    void operator()(const float *inputArray, float *resultArray);

  private:
    // メンバ変数の宣言
    std::size_t array_length_;
    std::unique_ptr<MetalFunctionBase> softmax_;
};

class MetalSigmoidFunction
{
  public:
    MetalSigmoidFunction(std::size_t arrayLength);
    void operator()(const float *inputArray, float *resultArray);

  private:
    // メンバ変数の宣言
    std::size_t array_length_;
    std::unique_ptr<MetalFunctionBase> sigmoid_;
};

class MetalReluFunction
{
  public:
    MetalReluFunction(std::size_t arrayLength);
    void operator()(const float *inputArray, float *resultArray);

  private:
    // メンバ変数の宣言
    std::size_t array_length_;
    std::unique_ptr<MetalFunctionBase> relu_;
};

class MetalMatMulFunction
{
  public:
    MetalMatMulFunction(std::size_t n, std::size_t m, std::size_t l);
    void operator()(const float *inputA, const float *inputB, float *result);

  private:
    // メンバ変数の宣言
    std::size_t n_;
    std::size_t m_;
    std::size_t l_;
    std::unique_ptr<MetalFunctionBase> matmul_;
};

class MetalDotProductFunction
{
  public:
    MetalDotProductFunction(std::size_t length);
    void operator()(const float *inputA, const float *inputB, float *result);

  private:
    std::size_t array_length_;
    std::unique_ptr<MetalFunctionBase> dot_product_;
};

class MetalAddArrayBatchFunction
{
  public:
    MetalAddArrayBatchFunction(std::size_t length, std::size_t batch_size);
    void operator()(const float *inputA, const float *inputB, float *result);

  private:
    std::size_t array_length_;
    std::unique_ptr<MetalFunctionBase> add_array_batch_;
    std::size_t batch_size_;
};

enum class ArithmeticType : std::size_t
{
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
  Sqrt = 4,
  Sum = 5,
  Softmax = 6,
  Sigmoid = 7,
  Relu = 8,
  DotProduct = 9,
  None = 10,
};
const std::string InvalidKernelFunctionName = "None";

const std::string KernelFunctionName[] = {
  "add_array_batch",
  "sub_array_batch",
  "mul_array_batch",
  "div_array_batch",
  InvalidKernelFunctionName,
  InvalidKernelFunctionName,
  InvalidKernelFunctionName,
  InvalidKernelFunctionName,
  InvalidKernelFunctionName,
  InvalidKernelFunctionName,
  InvalidKernelFunctionName,
  InvalidKernelFunctionName,
};

/**
 * @brief MetalArithmeticFunction
 *
 * このクラスは要素ごとの演算を行うクラスです.
 * 演算によっては入力バッファを2つ持つことも可能.
 * Sum などの演算は
 */
class MetalArithmeticFunction
{
  public:
    /**
     * @brief コンストラクタ
     * @param length 入力バッファの長さ. length > 0
     * @param batch_size バッチサイズ. batch_size > 0
     */
    MetalArithmeticFunction(std::size_t length, std::size_t batch_size);

    /**
     * @brief 入力バッファをセットする.
     * サイズは array_length_ * batch_size_ であると想定.
     * @param inputA 入力バッファA
     */
    void setInputA(const float *inputA);

    /**
     * @brief 入力バッファをセットする.
     * サイズは array_length_ * batch_size_ であると想定.
     * @param inputB 入力バッファB
     */
    void setInputB(const float *inputB);

    /**
     * @brief 計算結果を格納する
     * Sumなどの結果が一つの演算は, 1つのバッファサイズで良い.
     * バッファサイズの確認は内部で行っていない.
     * @param result 結果を格納するバッファ
     */
    void setResult(float *result);

    /**
     * @brief 演算を実行する.
     * @param arithmetic_type 演算の種類
     */
    void execute(ArithmeticType arithmetic_type);

  private:
    /**
     * ２つの値を元に演算を実行する.
     * 使用できない演算が指定された場合は例外を送出する.
     *
     * @param arithmetic_type 実行したいカネールの種類
     */
    void executeTwoValueOp(ArithmeticType arithmetic_type);



    std::string getKernelFunctionName(ArithmeticType arithmetic_type);

    std::size_t array_length_;
    std::size_t batch_size_;
    std::unique_ptr<MetalFunctionBase> arithmetic_;
    const float *input_a_;
    const float *input_b_;
    float *result_;
};
} // namespace nagato::mtl

#endif //NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_

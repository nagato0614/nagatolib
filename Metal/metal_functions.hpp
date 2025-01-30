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

}

#include "metal_functions_impl.hpp"

#endif //NAGATOLIB_METAL_METAL_FUNCTIONS_HPP_

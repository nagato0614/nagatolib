//
// Created by toru on 2024/08/31.
//

#include "metal_functions.hpp"
#include <chrono>

namespace nagato::mtl
{
auto &MLASingleton::GetMetalBase()
{
  return metal_base_;
}

std::unique_ptr<MetalFunctionBase> &MLASingleton::GetFunction(const std::string &function_name)
{
  try
  {
    auto &functions_ = GetInstance().functions_;
    auto &function = functions_.at(function_name);

    // 初期化しておく
    function->Reset();

    return function;
  } catch (const std::out_of_range &e)
  {
    throw std::runtime_error("Function not found: " + function_name);
  }
}

MLASingleton::MLASingleton()
{
  metal_base_ = std::make_unique<MetalBase>("../metal_kernel/linear_algebra.metal");

  this->GenerateAllKernelFunctions();
}

void MLASingleton::GenerateAllKernelFunctions()
{
  for (const auto &kernel_name : KernelFunctionNames)
  {
    // 登録されていない関数はスキップ
    if (kernel_name == InvalidKernelFunctionName)
    {
      continue;
    }
    auto &base = GetMetalBase();
    auto function = base->CreateFunctionBase(kernel_name);

    // 関数を登録
    this->functions_[kernel_name] = std::move(function);
  }
}

MetalAdderFunction::MetalAdderFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  add_arrays_ = base->CreateFunctionBase("add_arrays");
}

void MetalAdderFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result
)
{
  // buffer の作成
  auto bufferA = add_arrays_->CreateBufferFromHost(inA, buffer_length_);
  auto bufferB = add_arrays_->CreateBufferFromHost(inB, buffer_length_);
  auto bufferResult = add_arrays_->CreateBufferFromHost(result, buffer_length_);
  auto bufferLength = add_arrays_->CreateBuffer<uint>(1);
  auto buffer_length = static_cast<uint>(buffer_length_);

  // buffer にデータをコピー
  bufferLength.CopyToDevice(&buffer_length, 1);

  // buffer を function にセット
  add_arrays_->SetBuffer(bufferA, 0, 0);
  add_arrays_->SetBuffer(bufferB, 0, 1);
  add_arrays_->SetBuffer(bufferResult, 0, 2);
  add_arrays_->SetBuffer(bufferLength, 0, 3); // buffer_length をカーネルに渡す

  // スレッドグループサイズとグリッドサイズを設定
  const uint threadsPerGroup = DefaultThreadPerGroup;
  uint threadGroups =
    ((buffer_length_ + DataSizePerThread - 1) / DataSizePerThread + threadsPerGroup - 1)
    / threadsPerGroup;

  MTL::Size grid_size = MTL::Size(threadsPerGroup * threadGroups, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // 関数の実行
  add_arrays_->ExecuteKernel(grid_size, thread_group_size);
}

MetalSubFunction::MetalSubFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sub_arrays_ = base->CreateFunctionBase("sub_arrays");
}

void MetalSubFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result)
{
  // buffer の作成
  auto bufferA = sub_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = sub_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = sub_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);

  // buffer を function にセット
  sub_arrays_->SetBuffer(bufferA, 0, 0);
  sub_arrays_->SetBuffer(bufferB, 0, 1);
  sub_arrays_->SetBuffer(bufferResult, 0, 2);

  // 関数の実行
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  sub_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalMulFunction::MetalMulFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  mul_arrays_ = base->CreateFunctionBase("mul_arrays");
}

void MetalMulFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result)
{
  // buffer の作成
  auto bufferA = mul_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = mul_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = mul_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);

  // buffer を function にセット
  mul_arrays_->SetBuffer(bufferA, 0, 0);
  mul_arrays_->SetBuffer(bufferB, 0, 1);
  mul_arrays_->SetBuffer(bufferResult, 0, 2);

  // 関数の実行
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  mul_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalDivFunction::MetalDivFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  div_arrays_ = base->CreateFunctionBase("div_arrays");
}

void MetalDivFunction::operator()(
  const nFloat *inA,
  const nFloat *inB,
  nFloat *result)
{
  // buffer の作成
  auto bufferA = div_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferB = div_arrays_->CreateBuffer<float>(buffer_length_);
  auto bufferResult = div_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer にデータをコピー
  bufferA.CopyToDevice(inA, buffer_length_);
  bufferB.CopyToDevice(inB, buffer_length_);

  // buffer を function にセット
  div_arrays_->SetBuffer(bufferA, 0, 0);
  div_arrays_->SetBuffer(bufferB, 0, 1);
  div_arrays_->SetBuffer(bufferResult, 0, 2);

  // 関数の実行
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  div_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalSqrtFunction::MetalSqrtFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sqrt_arrays_ = base->CreateFunctionBase("sqrt_arrays");
}

void MetalSqrtFunction::operator()(const nFloat *inA, nFloat *result)
{
  // buffer の作成
  auto bufferA = sqrt_arrays_->CreateBufferFromHost(inA, buffer_length_);
  auto bufferResult = sqrt_arrays_->CreateBuffer<float>(buffer_length_);

  // buffer を function にセット
  sqrt_arrays_->SetBuffer(bufferA, 0, 0);
  sqrt_arrays_->SetBuffer(bufferResult, 0, 1);

  // スレッドグループサイズとグリッドサイズを設定
  const uint threadsPerGroup = DefaultThreadPerGroup;
  uint threadGroups = (buffer_length_ + threadsPerGroup - 1) / threadsPerGroup;

  MTL::Size grid_size = MTL::Size(threadsPerGroup * threadGroups, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // 関数の実行
  sqrt_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, buffer_length_);
}

MetalSumFunction::MetalSumFunction(std::size_t length)
  : buffer_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sum_arrays_ = base->CreateFunctionBase("sum_arrays");
}

void MetalSumFunction::operator()(const nFloat *inA, nFloat *result)
{
  // buffer の作成
  auto bufferA = sum_arrays_->CreateBufferFromHost(inA, buffer_length_);
  auto bufferResult = sum_arrays_->CreateBuffer<float>(1);
  auto buffer_params = sum_arrays_->CreateBuffer<unsigned int>(2);

  // 結果を初期化
  bufferResult[0] = 0.0f;
  buffer_params[0] = static_cast<unsigned int>(this->buffer_length_); // arraySize

  // buffer を function にセット
  sum_arrays_->SetBuffer(bufferA, 0, 0);
  sum_arrays_->SetBuffer(bufferResult, 0, 1);
  sum_arrays_->SetBuffer(buffer_params, 0, 2);

  // グループ共有メモリの確保
  //    1次元ディスパッチで、groupCount × threadsPerGroup = totalThreads
  const uint threadsPerGroup = 256;
  const uint groupCount = (buffer_length_ + threadsPerGroup - 1) / threadsPerGroup;
  uint totalThreads = groupCount * threadsPerGroup;
  buffer_params[1] = totalThreads; // totalThreads
  const size_t shared_memory_size = threadsPerGroup * sizeof(float);
  sum_arrays_->SetThreadgroupMemoryLength(shared_memory_size, 0);

  // 実行時のグリッドサイズ & スレッドグループサイズを設定
  MTL::Size grid_size = MTL::Size(totalThreads, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // 関数の実行
  sum_arrays_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, 1);
}

MetalSoftmaxFunction::MetalSoftmaxFunction(std::size_t arrayLength)
  : array_length_(arrayLength)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  softmax_ = base->CreateFunctionBase("softmax");
}

void MetalSoftmaxFunction::operator()(const float *inputArray, float *resultArray)
{
  // バッファの作成
  auto bufferInput = softmax_->CreateBufferFromHost(inputArray, array_length_);
  auto bufferResult = softmax_->CreateBuffer<float>(array_length_);
  auto bufferGlobalSum = softmax_->CreateBuffer<float>(1);
  auto bufferArraySize = softmax_->CreateBuffer<uint>(1);

  bufferArraySize[0] = static_cast<uint>(array_length_);

  // バッファを関数にセット
  softmax_->SetBuffer(bufferInput, 0, 0);
  softmax_->SetBuffer(bufferResult, 0, 1);
  softmax_->SetBuffer(bufferGlobalSum, 0, 2);
  softmax_->SetBuffer(bufferArraySize, 0, 3);

  // スレッドグループサイズとグリッドサイズを設定
  const uint threadsPerGroup = 256;
  const uint groupCount = (array_length_ + threadsPerGroup - 1) / threadsPerGroup;
  uint totalThreads = groupCount * threadsPerGroup;
  bufferArraySize[1] = totalThreads;

  MTL::Size grid_size = MTL::Size(totalThreads, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // グループ共有メモリの確保
  const size_t shared_memory_size = threadsPerGroup * sizeof(float);
  softmax_->SetThreadgroupMemoryLength(shared_memory_size, 0);

  // 関数の実行
  softmax_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(resultArray, array_length_);
}

MetalSigmoidFunction::MetalSigmoidFunction(std::size_t arrayLength)
  : array_length_(arrayLength)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  sigmoid_ = base->CreateFunctionBase("sigmoid_array");
}

void MetalSigmoidFunction::operator()(const float *inputArray, float *resultArray)
{
  // バッファの作成
  auto bufferInput = sigmoid_->CreateBuffer<float>(array_length_);
  auto bufferResult = sigmoid_->CreateBuffer<float>(array_length_);

  // バッファにデータをコピー
  bufferInput.CopyToDevice(inputArray, array_length_);

  // バッファを関数にセット
  sigmoid_->SetBuffer(bufferInput, 0, 0);
  sigmoid_->SetBuffer(bufferResult, 0, 1);

  // 実行
  MTL::Size grid_size = MTL::Size(array_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  sigmoid_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(resultArray, array_length_);
}

MetalReluFunction::MetalReluFunction(std::size_t arrayLength)
  : array_length_(arrayLength)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  relu_ = base->CreateFunctionBase("relu_array");
}

void MetalReluFunction::operator()(
  const float *inputArray,
  float *resultArray
)
{
  // バッファの作成
  auto bufferInput = relu_->CreateBuffer<float>(array_length_);
  auto bufferResult = relu_->CreateBuffer<float>(array_length_);
  auto bufferArraySize = relu_->CreateBuffer<uint>(1);

  // バッファにデータをコピー
  bufferInput.CopyToDevice(inputArray, array_length_);
  bufferArraySize[0] = static_cast<uint>(array_length_);

  // バッファを関数にセット
  relu_->SetBuffer(bufferInput, 0, 0);
  relu_->SetBuffer(bufferResult, 0, 1);
  relu_->SetBuffer(bufferArraySize, 0, 2);

  // 実行
  MTL::Size grid_size = MTL::Size(array_length_, 1, 1);
  MTL::Size thread_group_size = MTL::Size(DefaultThreadPerGroup, 1, 1);
  relu_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(resultArray, array_length_);
}

MetalMatMulFunction::MetalMatMulFunction(std::size_t n, std::size_t m, std::size_t l)
  : n_(n), m_(m), l_(l)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  matmul_ = base->CreateFunctionBase("matmul_array");
}

void MetalMatMulFunction::operator()(const float *inputA, const float *inputB, float *result)
{
  // バッファの作成
  auto bufferA = matmul_->CreateBufferFromHost(inputA, n_ * m_);
  auto bufferB = matmul_->CreateBufferFromHost(inputB, m_ * l_);
  auto bufferResult = matmul_->CreateBuffer<float>(n_ * l_);
  auto bufferN = matmul_->CreateBuffer<uint>(1);
  auto bufferM = matmul_->CreateBuffer<uint>(1);
  auto bufferL = matmul_->CreateBuffer<uint>(1);

  // 定数をセット
  bufferN[0] = static_cast<uint>(n_);
  bufferM[0] = static_cast<uint>(m_);
  bufferL[0] = static_cast<uint>(l_);

  // バッファを関数にセット
  matmul_->SetBuffer(bufferA, 0, 0);
  matmul_->SetBuffer(bufferB, 0, 1);
  matmul_->SetBuffer(bufferResult, 0, 2);
  matmul_->SetBuffer(bufferN, 0, 3);
  matmul_->SetBuffer(bufferM, 0, 4);
  matmul_->SetBuffer(bufferL, 0, 5);

  // 実行
  MTL::Size grid_size = MTL::Size(n_, l_, 1);
  MTL::Size thread_group_size = MTL::Size(16, 16, 1);
  matmul_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, n_ * l_);
}

MetalDotProductFunction::MetalDotProductFunction(std::size_t length)
  : array_length_(length)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  dot_product_ = base->CreateFunctionBase("dot_product");
}

void MetalDotProductFunction::operator()(const float *inputA, const float *inputB, float *result)
{
  // バッファの作成
  auto bufferA = dot_product_->CreateBufferFromHost(inputA, array_length_);
  auto bufferB = dot_product_->CreateBufferFromHost(inputB, array_length_);
  auto bufferResult = dot_product_->CreateBuffer<float>(1);
  auto bufferLength = dot_product_->CreateBuffer<uint>(1);

  // 定数をセット
  bufferLength[0] = static_cast<uint>(array_length_);

  // バッファを関数にセット
  dot_product_->SetBuffer(bufferA, 0, 0);
  dot_product_->SetBuffer(bufferB, 0, 1);
  dot_product_->SetBuffer(bufferResult, 0, 2);
  dot_product_->SetBuffer(bufferLength, 0, 3);

  // スレッドグループサイズとグリッドサイズを設定
  const uint threadsPerGroup = 256;
  const uint groupCount = (array_length_ + threadsPerGroup - 1) / threadsPerGroup;
  uint totalThreads = groupCount * threadsPerGroup;

  MTL::Size grid_size = MTL::Size(totalThreads, 1, 1);
  MTL::Size thread_group_size = MTL::Size(threadsPerGroup, 1, 1);

  // グループ共有メモリの確保
  const size_t shared_memory_size = threadsPerGroup * sizeof(float);
  dot_product_->SetThreadgroupMemoryLength(shared_memory_size, 0);

  // 関数の実行
  dot_product_->ExecuteKernel(grid_size, thread_group_size);

  // 結果をコピー
  bufferResult.CopyToHost(result, 1);
}

MetalAddArrayBatchFunction::MetalAddArrayBatchFunction(std::size_t length, std::size_t batch_size)
  : array_length_(length), batch_size_(batch_size)
{
  auto &base = MLASingleton::GetInstance().GetMetalBase();
  add_array_batch_ = base->CreateFunctionBase("add_array_batch");
}

void MetalAddArrayBatchFunction::operator()(const float *inputA, const float *inputB, float *result)
{
  const std::size_t buffer_length = array_length_ * batch_size_;

  // バッファの作成
  auto bufferA = add_array_batch_->CreateBufferFromHost(inputA, buffer_length);
  auto bufferB = add_array_batch_->CreateBufferFromHost(inputB, buffer_length);
  auto bufferResult = add_array_batch_->CreateBufferFromHost(result, buffer_length);
  auto bufferLength = add_array_batch_->CreateBuffer<uint>(1);
  auto bufferBatchSize = add_array_batch_->CreateBuffer<uint>(1);

  // 定数をセット
  bufferLength[0] = static_cast<uint>(array_length_);
  bufferBatchSize[0] = static_cast<uint>(batch_size_);

  // バッファを関数にセット
  add_array_batch_->SetBuffer(bufferA, 0, 0);
  add_array_batch_->SetBuffer(bufferB, 0, 1);
  add_array_batch_->SetBuffer(bufferResult, 0, 2);
  add_array_batch_->SetBuffer(bufferLength, 0, 3);
  add_array_batch_->SetBuffer(bufferBatchSize, 0, 4);

  // 実行
  // 1スレッドが担う「配列要素数」のかたまり分だけ、x方向にスレッドを立てる
  // バッチサイズはz方向に立てる
  const uint threads_per_batch = std::ceil(
    static_cast<double>(array_length_) / static_cast<double>(DataSizePerThread)
  );
  std::cout << "threads_per_batch: " << threads_per_batch << std::endl;

  // z方向 = バッチサイズ (batch_size_)
  MTL::Size threads_per_grid = MTL::Size(threads_per_batch,
                                         1,
                                         batch_size_);

  MTL::Size thread_per_threadgroup = MTL::Size(16, 16, 1);
  add_array_batch_->ExecuteKernel(threads_per_grid, thread_per_threadgroup);
}

MetalArithmeticFunction::MetalArithmeticFunction(std::size_t length, std::size_t batch_size)
  : array_length_(length),
    batch_size_(batch_size),
    input_a_(nullptr),
    input_b_(nullptr),
    result_(nullptr)
{
}
void MetalArithmeticFunction::setInputA(const float *inputA)
{
  this->input_a_ = inputA;
}
void MetalArithmeticFunction::setInputB(const float *inputB)
{
  this->input_b_ = inputB;
}
void MetalArithmeticFunction::setResult(float *result)
{
  this->result_ = result;
}

void MetalArithmeticFunction::execute(ArithmeticType arithmetic_type)
{
  switch (arithmetic_type)
  {
    case ArithmeticType::Add:
    case ArithmeticType::Sub:
    case ArithmeticType::Mul:
    case ArithmeticType::Div:
      this->executeTwoValueOp(arithmetic_type);
      break;

    // 以下, 未実装
    case ArithmeticType::Sqrt:
      this->executeOneValueOp(arithmetic_type);
      break;
    case ArithmeticType::Sum:
    case ArithmeticType::Softmax:
    case ArithmeticType::Sigmoid:
    case ArithmeticType::Relu:
    case ArithmeticType::DotProduct:
    case ArithmeticType::None:
      break;
  }
}

void MetalArithmeticFunction::executeTwoValueOp(ArithmeticType arithmetic_type)
{
  // 四則演算以外は例外を創出する
  if (
    arithmetic_type != ArithmeticType::Add &&
    arithmetic_type != ArithmeticType::Sub &&
    arithmetic_type != ArithmeticType::Mul &&
    arithmetic_type != ArithmeticType::Div
  )
  {
    throw std::invalid_argument("Not Found Arithmetic Type");
  }

  auto &base = MLASingleton::GetInstance().GetMetalBase();
  const std::string kernel_function_name = getKernelFunctionName(arithmetic_type);
  auto &arithmetic_ = MLASingleton::GetInstance().GetFunction(kernel_function_name);
  const std::size_t buffer_length = array_length_ * batch_size_;

  // データが割り当てられているか確認
  if (this->input_a_ == nullptr)
  {
    throw std::invalid_argument("input_a_ is nullptr");
  }
  if (this->input_b_ == nullptr)
  {
    throw std::invalid_argument("input_b_ is nullptr");
  }
  if (this->result_ == nullptr)
  {
    throw std::invalid_argument("result_ is nullptr");
  }

  // バッファの作成
  auto bufferA = arithmetic_->CreateBufferFromHost(this->input_a_, buffer_length);
  auto bufferB = arithmetic_->CreateBufferFromHost(this->input_b_, buffer_length);
  auto bufferResult = arithmetic_->CreateBufferFromHost(this->result_, buffer_length);
  auto bufferLength = arithmetic_->CreateBuffer<uint>(1);
  auto bufferBatchSize = arithmetic_->CreateBuffer<uint>(1);

  // 定数をセット
  bufferLength[0] = static_cast<uint>(array_length_);
  bufferBatchSize[0] = static_cast<uint>(batch_size_);

  // バッファを関数にセット
  arithmetic_->SetBuffer(bufferA, 0, 0);
  arithmetic_->SetBuffer(bufferB, 0, 1);
  arithmetic_->SetBuffer(bufferResult, 0, 2);
  arithmetic_->SetBuffer(bufferLength, 0, 3);
  arithmetic_->SetBuffer(bufferBatchSize, 0, 4);

  // 実行
  const uint threads_per_batch = std::ceil(
    static_cast<double>(array_length_) / static_cast<double>(DataSizePerThread)
  );
  const auto grid_size = MTL::Size(threads_per_batch, 1, batch_size_);
  const auto thread_per_threadgroup = MTL::Size(16, 16, 1);

  // 時間計測
  arithmetic_->ExecuteKernel(grid_size, thread_per_threadgroup);
}

void MetalArithmeticFunction::executeOneValueOp(ArithmeticType arithmetic_type)
{
  // Sum, Sigmoid, Relu のみ対応
  if (
    arithmetic_type != ArithmeticType::Sigmoid &&
    arithmetic_type != ArithmeticType::Relu &&
    arithmetic_type != ArithmeticType::Sqrt
  )
  {
    throw std::invalid_argument("Not Found Arithmetic Type");
  }

  auto &base = MLASingleton::GetInstance().GetMetalBase();
  const std::string kernel_function_name = getKernelFunctionName(arithmetic_type);
  auto &arithmetic_ = MLASingleton::GetInstance().GetFunction(kernel_function_name);
  const std::size_t buffer_length = array_length_ * batch_size_;

  // データが割り当てられているか確認
  if (this->input_a_ == nullptr)
  {
    throw std::invalid_argument("input_a_ is nullptr");
  }
  if (this->result_ == nullptr)
  {
    throw std::invalid_argument("result_ is nullptr");
  }

  // バッファの作成
  auto bufferA = arithmetic_->CreateBufferFromHost(this->input_a_, buffer_length);
  auto bufferResult = arithmetic_->CreateBufferFromHost(this->result_, buffer_length);
  auto bufferLength = arithmetic_->CreateBuffer<uint>(1);
  auto bufferBatchSize = arithmetic_->CreateBuffer<uint>(1);

  // 定数をセット
  bufferLength[0] = static_cast<uint>(array_length_);
  bufferBatchSize[0] = static_cast<uint>(batch_size_);

  // バッファを関数にセット
  arithmetic_->SetBuffer(bufferA, 0, 0);
  arithmetic_->SetBuffer(bufferResult, 0, 1);
  arithmetic_->SetBuffer(bufferLength, 0, 2);
  arithmetic_->SetBuffer(bufferBatchSize, 0, 3);

  // 実行
  const uint threads_per_batch = std::ceil(
    static_cast<double>(array_length_) / static_cast<double>(DataSizePerThread)
  );
  const auto grid_size = MTL::Size(threads_per_batch, 1, batch_size_);
  const auto thread_per_threadgroup = MTL::Size(16, 16, 1);

  arithmetic_->ExecuteKernel(grid_size, thread_per_threadgroup);
}

std::string MetalArithmeticFunction::getKernelFunctionName(ArithmeticType arithmetic_type)
{
  std::string ret = InvalidKernelFunctionName;
  switch (arithmetic_type)
  {
    case ArithmeticType::Add:
      ret = KernelFunctionNames[static_cast<std::size_t>(ArithmeticType::Add)];
      break;
    case ArithmeticType::Sub:
      ret = KernelFunctionNames[static_cast<std::size_t>(ArithmeticType::Sub)];
      break;
    case ArithmeticType::Mul:
      ret = KernelFunctionNames[static_cast<std::size_t>(ArithmeticType::Mul)];
      break;
    case ArithmeticType::Div:
      ret = KernelFunctionNames[static_cast<std::size_t>(ArithmeticType::Div)];
      break;

    // 以下, 未実装
    case ArithmeticType::Sqrt:
      ret = KernelFunctionNames[static_cast<std::size_t>(ArithmeticType::Sqrt)];
      break;
    case ArithmeticType::Sum:
    case ArithmeticType::Softmax:
    case ArithmeticType::Sigmoid:
    case ArithmeticType::Relu:
    case ArithmeticType::DotProduct:
    case ArithmeticType::None:
      break;
  }

  if (ret == InvalidKernelFunctionName)
  {
    throw std::invalid_argument("Not Found Kernel Function Name");
  }

  return ret;
}
} // namespace nagato::mtl

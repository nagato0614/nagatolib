//
// Created by toru on 2024/08/29.
//

#ifndef NAGATOLIB_SRC_METAL_BASE_HPP_
#define NAGATOLIB_SRC_METAL_BASE_HPP_

// c++ standard library
#include <string>

// metal
#include "Metal.hpp"

// nagato metal library
#include "metal_buffer.hpp"

namespace nagato::mtl
{
class MetalFunctionBase
{
 public:
  MetalFunctionBase(std::size_t buffer_length,
                    MTL::ComputePipelineState *function_pso,
                    MTL::Device *p_device,
                    MTL::CommandQueue *p_command_queue);

  ~MetalFunctionBase();

  /**
   * buffer を　encoder にセットする
   */
  void SetBuffer(const MTL::Buffer *buffer,
                 int offset,
                 int index);

  template<typename T>
  void SetBuffer(const MetalBuffer<T> &buffer,
                 int offset,
                 int index)
  {
    SetBuffer(buffer.GetBuffer(), offset, index);
  }

  [[nodiscard]] std::size_t maxTotalThreadsPerThreadgroup() const;

  /**
 * Buffer を作成する.
 * @tparam T バッファの型
 * @param buffer_length　バッファの長さ. デフォルトは1
 * @return
 */
  template<typename T>
  [[nodiscard]] MetalBuffer<T> CreateBuffer(std::size_t buffer_length = 0)
  {
    auto l = buffer_length_;
    if (buffer_length != 0)
    {
      l = buffer_length;
    }
    MetalBuffer<T> buffer(device_.get(), l);
    return buffer;
  }

  template<typename T>
  void SetConstant(const T &value, std::size_t length, int index)
  {
    this->compute_command_encoder_->setBytes(&value, length * sizeof(T), index);
  }

  /**
   * リセットして再度カーネルを実行できる状態にする.
   */
  void Reset();

  /**
   * カーネルを実行する
   * 先にバッファをセットしておく必要がある
   */
  void ExecuteKernel();

  void ExecuteKernel(MTL::Size grid_size, MTL::Size thread_group_size);

 private:
  std::string kernel_file_name_;
  std::string kernel_function_name_;
  std::size_t buffer_length_;
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  NS::SharedPtr<MTL::ComputePipelineState> function_pso_;
  NS::SharedPtr<MTL::CommandBuffer> command_buffer_;
  NS::SharedPtr<MTL::ComputeCommandEncoder> compute_command_encoder_;
};
}
#endif // NAGATOLIB_SRC_METAL_BASE_HPP_

//
// Created by toru on 2024/08/29.
//

#ifndef NAGATOLIB_SRC_METAL_BASE_HPP_
#define NAGATOLIB_SRC_METAL_BASE_HPP_

#include <string>
#include "Metal.hpp"
#include "metal_buffer.hpp"

class MetalFunctionBase
{
 public:
  MetalFunctionBase(std::string kernel_file_name,
                    std::string kernel_function_name,
                    std::size_t buffer_length,
                    MTL::Device *p_device,
                    MTL::CommandQueue *p_command_queue);

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

  /**
   * リセットして再度カーネルを実行できる状態にする.
   */
  void Reset();

  /**
   * カーネルを実行する
   * 先にバッファをセットしておく必要がある
   */
  void ExecuteKernel();

  void Release();
 private:
  std::string kernel_file_name_;
  std::string kernel_function_name_;
  std::size_t buffer_length_;
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::Function> kernel_function_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  NS::SharedPtr<MTL::ComputePipelineState> function_pso_;
  NS::SharedPtr<MTL::CommandBuffer> command_buffer_;
  NS::SharedPtr<MTL::ComputeCommandEncoder> compute_command_encoder_;
};

#endif // NAGATOLIB_SRC_METAL_BASE_HPP_

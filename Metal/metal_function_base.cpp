//
// Created by toru on 2024/08/29.
//


#include "metal_function_base.hpp"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <utility>

MetalFunctionBase::MetalFunctionBase(std::string kernel_file_name,
                                     std::string kernel_function_name,
                                     std::size_t buffer_length,
                                     MTL::Device *p_device,
                                     MTL::CommandQueue *p_command_queue) :
  kernel_file_name_(std::move(kernel_file_name)),
  kernel_function_name_(std::move(kernel_function_name)),
  buffer_length_(buffer_length),
  device_(NS::RetainPtr(p_device)),
  command_queue_(NS::RetainPtr(p_command_queue))
{
  NS::Error *error = nullptr;

  const auto librarySource = [this]
  {
    std::cout << "Kernel file name: " << this->kernel_file_name_ << std::endl;
    std::ifstream source(this->kernel_file_name_);
    return std::string((std::istreambuf_iterator<char>(source)), {});
  }();

  NS::SharedPtr<MTL::Library> library =
    NS::TransferPtr(
      device_->newLibrary(
        NS::String::string(
          librarySource.c_str(),
          NS::ASCIIStringEncoding),
        nullptr,
        &error)
    );

  if (library.get() == nullptr || error != nullptr)
  {
    throw std::runtime_error("Failed to create Metal library.");
  }

  std::cout << "Kernel function name: " << kernel_function_name_ << std::endl;
  kernel_function_ = NS::TransferPtr(
    library->newFunction(
      NS::String::string(
        kernel_function_name_.c_str(),
        NS::ASCIIStringEncoding)
    )
  );

  if (!kernel_function_)
  {
    std::cerr << "Failed to find the kernel function." << std::endl;
    exit(1);
  }

  function_pso_ = NS::TransferPtr(
    device_->newComputePipelineState(
      kernel_function_->retain(),
      &error
    )
  );

  if (function_pso_.get() == nullptr || error != nullptr)
  {
    throw std::runtime_error("Failed to create Metal compute pipeline state.");
  }
  command_buffer_ = NS::RetainPtr(command_queue_->commandBuffer());
  compute_command_encoder_ = NS::RetainPtr(command_buffer_->computeCommandEncoder());
  compute_command_encoder_->setComputePipelineState(function_pso_->retain());
}

void MetalFunctionBase::ExecuteKernel()
{
  MTL::Size grid_size = MTL::Size(buffer_length_, 1, 1);

  NS::UInteger thread_group_size_ = function_pso_->maxTotalThreadsPerThreadgroup();
  if (thread_group_size_ > buffer_length_)
  {
    thread_group_size_ = buffer_length_;
  }

  MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

  compute_command_encoder_->dispatchThreads(grid_size, thread_group_size);

  compute_command_encoder_->endEncoding();
  command_buffer_->commit();
  command_buffer_->waitUntilCompleted();
}

void MetalFunctionBase::SetBuffer(const MTL::Buffer *buffer, int offset, int index)
{
  compute_command_encoder_->setBuffer(buffer, offset, index);
}

void MetalFunctionBase::Reset()
{
  // 以前のコマンドエンコーダを終了して解放
  if (compute_command_encoder_)
  {
    compute_command_encoder_->endEncoding();
    compute_command_encoder_.reset();
  }

  // 以前のコマンドバッファを解放
  if (command_buffer_)
  {
    command_buffer_.reset();
  }

  // 新しいコマンドバッファとコマンドエンコーダを作成
  command_buffer_ = NS::RetainPtr(command_queue_->commandBuffer());
  compute_command_encoder_ = NS::RetainPtr(command_buffer_->computeCommandEncoder());
  compute_command_encoder_->setComputePipelineState(function_pso_->retain());
}
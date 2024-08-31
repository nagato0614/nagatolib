//
// Created by toru on 2024/08/31.
//

#include "metal_base.hpp"

#include <utility>

namespace nagato
{

MetalBase::MetalBase()
{
  pool_ = NS::AutoreleasePool::alloc()->init();
  device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  command_queue_ = NS::TransferPtr(device_->newCommandQueue());
}

MetalBase::~MetalBase()
{
  pool_->release();
}


MetalFunctionBase MetalBase::CreateFunctionBase(std::string kernel_file_name,
                                                std::string kernel_function_name,
                                                std::size_t buffer_length)
{
  MetalFunctionBase metal_function_base(std::move(kernel_file_name),
                                        std::move(kernel_function_name),
                                        buffer_length,
                                        device_.get(),
                                        command_queue_.get());
  return metal_function_base;
}

}

//
// Created by toru on 2024/08/31.
//

#include "metal_base.hpp"

#include <utility>
#include <fstream>
#include <memory>

namespace nagato
{

MetalBase::MetalBase(std::string kernel_file_name)
  : kernel_file_name_(std::move(kernel_file_name))
{
  pool_ = NS::AutoreleasePool::alloc()->init();
  device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  command_queue_ = NS::TransferPtr(device_->newCommandQueue());

  // kernel ファイルを読み込みライブラリを作成する
  NS::Error *error = nullptr;

  const auto librarySource = [this]
  {
    std::cout << "Kernel file name: " << this->kernel_file_name_ << std::endl;
    std::ifstream source(this->kernel_file_name_);
    return std::string((std::istreambuf_iterator<char>(source)), {});
  }();

  this->library_ =
    NS::TransferPtr(
      device_->newLibrary(
        NS::String::string(
          librarySource.c_str(),
          NS::ASCIIStringEncoding),
        nullptr,
        &error)
    );

  if (library_.get() == nullptr || error != nullptr)
  {
    if (error != nullptr)
    {
      std::cout << "Error: " << error->localizedDescription()->utf8String() << std::endl;
    }
    throw std::runtime_error("Failed to create Metal library.");
  }
}

MetalBase::~MetalBase()
{
  pool_->release();
}

std::unique_ptr<MetalFunctionBase> MetalBase::CreateFunctionBase(std::string kernel_function_name,
                                                                 std::size_t buffer_length)
{
  NS::Error *error = nullptr;

  auto kernel_function =
    NS::TransferPtr(
      library_->newFunction(
        NS::String::string(
          kernel_function_name.c_str(),
          NS::ASCIIStringEncoding)
      )
    );

  if (!kernel_function)
  {
    throw std::runtime_error("Failed to find the kernel function.");
  }

  auto function_pso = device_->newComputePipelineState(
    kernel_function->retain(),
    &error
  );

  if (function_pso == nullptr || error != nullptr)
  {
    throw std::runtime_error("Failed to create Metal compute pipeline state.");
  }

  auto metal_function_base = std::make_unique<MetalFunctionBase>(
    buffer_length,
    function_pso,
    device_.get(),
    command_queue_.get()
  );
  return metal_function_base;
}

}

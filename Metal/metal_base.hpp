//
// Created by toru on 2024/08/31.
//

#ifndef NAGATOLIB_METAL_METAL_BASE_HPP_
#define NAGATOLIB_METAL_METAL_BASE_HPP_

#include "Metal.hpp"
#include "metal_buffer.hpp"
#include "metal_function_base.hpp"

namespace nagato::mtl
{
class MetalBase
{
 public:
  explicit MetalBase(std::string kernel_file_name);

  ~MetalBase();

  /**
   * デバイスを取得. 所有権は呼び出し側にある
   * @return
   */
  [[nodiscard]] MTL::Device *GetDevice() const
  {
    return device_->retain();
  }

  /**
   * コマンドキューを取得. 所有権は呼び出し側にある
   * @return
   */
  [[nodiscard]] MTL::CommandQueue *GetCommandQueue() const
  {
    return command_queue_->retain();
  }



  [[nodiscard]] std::unique_ptr<MetalFunctionBase> CreateFunctionBase(std::string kernel_function_name,
                                                                      std::size_t buffer_length);

 private:
  std::string kernel_file_name_;
  NS::AutoreleasePool *pool_;
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> command_queue_;
  NS::SharedPtr<MTL::Library> library_;
};

} // namespace nagato
#endif //NAGATOLIB_METAL_METAL_BASE_HPP_

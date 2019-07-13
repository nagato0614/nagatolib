//
// Created by nagato0614 on 2019-07-13.
//

#ifndef NAGATOLIB_SRC_BLOCKED_QUEUE_HPP_
#define NAGATOLIB_SRC_BLOCKED_QUEUE_HPP_

#include <queue>
#include <mutex>

namespace nagato {

template<typename T>
class BlockedQueue {

  using _size = std::size_t;
  using _queue = std::queue<T>;
  using _lock = std::lock_guard<std::mutex>;
 public:
  BlockedQueue() = default;

  bool Empty() const noexcept {
    _lock lock(mutex_);
    return queue_.empty();
  }

  _size Size() const noexcept {
	_lock lock(mutex_);
	return queue_.size();
  }

  T& Front() noexcept {
	_lock lock(mutex_);
	return queue_.front;
  }

  const T& Front() const noexcept {
	_lock lock(mutex_);
	return queue_.front();
  }

  T& Back() noexcept {
	_lock lock(mutex_);
    return queue_.back();
  }

  const T& Back() const noexcept {
	_lock lock(mutex_);
    return queue_.back();
  }

  void Push(const T& x) noexcept {
	_lock lock(mutex_);
    queue_.push(std::forward<T&>(x));
  }

  void Push(T&& x) noexcept {
	_lock lock(mutex_);
    queue_.push(std::forward<T&&>(x));
  }

  template<class... Args>
  decltype(auto) Emplace(Args&&... args) noexcept {
	_lock lock(mutex_);
    return queue_.emplace_back(std::forward<Args>(args)...);
  }

  void Pop() noexcept {
	_lock lock(mutex_);
    queue_.pop();
  }

  void Swap(_queue& q) noexcept {
	_lock lock(mutex_);
	queue_.swap(q);
  }

 private:
  _queue queue_;
  std::mutex mutex_;
};
}
#endif //NAGATOLIB_SRC_BLOCKED_QUEUE_HPP_

//
// Created by nagato0614 on 2019-07-12.
//

#include "thread.hpp"

namespace nagato
{
ThreadPool::ThreadPool(size nthreads) noexcept
	: nthreads(nthreads)
{
  assert(nthreads > 0);

  for (size i = 0; i < nthreads; i++)
  {
	threads_.emplace_back(
		std::thread(
			[this]()
			{
			  while (true)
			  {
				unique_lock_t lock(mutex_);
				condition_variable_.wait(lock,
										 [this]()
										 {
										   return
											   (is_work_
												   && !queue_.empty())
												   || !is_work_;
										 });

				if (!is_work_ && queue_.empty())
				  return;

				process_t
					work = std::move(queue_.front());
				queue_.pop();
				lock.unlock();

				work();
			  }
			}));
  }
}

ThreadPool::~ThreadPool() noexcept
{
  Finish();
}

void ThreadPool::Loop(std::function<void(size)> &&f,
					  ThreadPool::size size) noexcept
{
  std::vector<std::future<void>> vector;

  for (ThreadPool::size i = 0; i < size; i++)
  {
	vector.push_back(EnqueuTask(f, i));
  }

  condition_variable_.notify_all();

  for (auto &i : vector)
  {
	if (i.valid())
	  i.get();
  }
}

void ThreadPool::Finish() noexcept
{
  is_work_ = false;
  ThreadPool::size joinable = 0;
  condition_variable_.notify_all();

  // すべてのスレッドが終了するまでjoinを行う
  do
  {
	joinable = 0;
	for (auto &thread : threads_)
	  if (thread.joinable())
		thread.join();
	  else
		joinable++;
  } while (joinable != nthreads);
}
}

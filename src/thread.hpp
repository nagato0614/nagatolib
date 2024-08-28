//
// Created by nagato0614 on 2019-07-12.
//

#ifndef NAGATOLIB_SRC_THREAD_HPP_
#define NAGATOLIB_SRC_THREAD_HPP_

#include <memory>
#include <thread>
#include <future>
#include <condition_variable>
#include <queue>
#include <cassert>
#include <functional>

#include "assert.hpp"

namespace nagato {
/**
 * https://vorbrodt.blog/2019/02/12/simple-thread-pool/
 * http://progsch.net/wordpress/?p=81
 * タスクを受け取って実行結果を受け取るfutureを返す
 */
class ThreadPool {
	using thread_pool = std::vector<std::thread>;
	using size = std::size_t;
	using process = std::function<void(void)>;
	using queue = std::queue<process>;
	using unique_lock = std::unique_lock<std::mutex>;

 public:

	explicit
	ThreadPool(
			size nthreads = std::thread::hardware_concurrency()
	) noexcept;

	~ThreadPool() noexcept;

	/**
	 * 受け取ったworkをqueueに追加する
	 * @tparam F
	 * @tparam Args
	 * @param f
	 * @param args
	 */
	template<typename F, typename... Args>
	void EnqueuWork(F &&f, Args &&... args) noexcept {
		unique_lock lock(mutex_);
		queue_.push([=]() { f(args...); });
		condition_variable_.notify_all();
	}

	/**
	 * 受け取ったタスクをqueueに追加する
	 * futureを返す
	 * @tparam F
	 * @tparam Args
	 * @param f
	 * @param args
	 * @return
	 */
	template<typename F, typename... Args>
	auto EnqueuTask(F &&f, Args &&...args)
	-> std::future<std::invoke_result_t<F, Args...>> {
		using return_type = std::invoke_result_t<F, Args...>;

		auto task
				= std::make_shared<std::packaged_task<return_type()>>(
						std::bind(std::forward<F>(f), std::forward<Args>(args)...)
				);

		std::future<return_type> res
				= task->get_future();
		{
			unique_lock lock(mutex_);
			queue_.push([task]() { (*task)(); });
		}
		condition_variable_.notify_all();
		return res;
	}

	/**
	 * 指定した回数fを実行する
	 * @param f
	 * @param size
	 */
	void Loop(std::function<void(size)> &&f, size size) noexcept;

	void Finish() noexcept;

 private:
	thread_pool threads_;
	queue queue_;
	bool is_work_ = true;
	size n_threads;
	std::mutex mutex_;
	std::condition_variable condition_variable_;
};
}
#endif //NAGATOLIB_SRC_THREAD_HPP_

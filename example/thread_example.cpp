//
// Created by nagato0614 on 2019-07-13.
//

#include <iostream>
#include <mutex>
#include <cstdlib>
#include <random>
#include "thread.hpp"
#include "vector.hpp"

std::mutex cout_lock;

template<typename T>
void trace(T x)
{
  std::scoped_lock<std::mutex> lock(cout_lock);
  std::cout << x << std::endl;
  std::fflush(stdout);
}

const int COUNT = std::thread::hardware_concurrency();
const int WORK = 10'000'000;

using namespace nagato;
int main()
{
  using vector3d = Vector<int, 3>;
  const int N = 100;
  ThreadPool pool;

  // 結果を保存
  std::vector<vector3d> result(N);

  std::cout << "work" << std::endl;
  // work
  for (int i = 0; i < N; i++)
  {
    auto work = [&result](int i)
    {
      vector3d v1{i, i, i};
      vector3d v2{i, i, i};

      result.at(i) = v1 + v2;
    };

    pool.EnqueuWork(work, i);
  }

  std::cout << "task" << std::endl;
  // task
  std::vector<std::future<void>> vector;
  for (int i = 0; i < N; i++)
  {
    auto task = [](int i)
    {
      vector3d v1{i, i, i};
      vector3d v2{i, i, i};

      std::cout << v1 + v2;
    };

    vector.push_back(
      pool.EnqueuTask(std::move(task), i)
    );
  }

  for (auto &i : vector)
  {
    i.get();
  }

  std::fflush(stdout);
  srand((unsigned int) time(nullptr));
  std::cout << "worker test" << std::endl;
  for (int i = 1; i <= COUNT; ++i)
    pool.EnqueuWork([](int worker_number)
                    {
                      int work_output = 0;
                      int work = WORK + (rand() % (WORK));
                      std::cout << "work item " + std::to_string(worker_number)
                        + " starting "
                        + std::to_string(work)
                        + " iterations..." << std::endl;
                      for (int w = 0; w < work; ++w)
                        work_output += rand();
                      std::cout
                        << "work item " + std::to_string(worker_number) + " finished"
                        << std::endl;
                    }, i);

  // ここで全てのワーカーが終了するまで待つ
  pool.Loop([](int i)
            {
              vector3d v1{i, i, i};
              vector3d v2{i, i, i};

              std::cout << v1 + v2;
            }, N);

  return 0;
}


//
// Created by toru on 2025/02/08.
//

#include "timer.hpp"
#include <iostream>
#include <vector>
Timer::Timer()
{
  // map を初期化
  timers_.clear();
}

Timer::~Timer()
{
}

Timer &Timer::GetInstance()
{
  static Timer instance;
  return instance;
}

void Timer::Start(const std::string &name)
{
  timers_[name].start_time = std::chrono::high_resolution_clock::now();
}

void Timer::Stop(const std::string &name)
{
  timers_[name].end_time = std::chrono::high_resolution_clock::now();
  timers_[name].duration = std::chrono::duration_cast<std::chrono::microseconds>(timers_[name].end_time - timers_[name].start_time);
}

void Timer::PrintDuration(const std::string &name)
{
  std::cout << name << " : " << timers_[name].duration.count() << " us" << std::endl;
}

std::chrono::microseconds Timer::GetDuration(const std::string &name)
{
  return timers_[name].duration;
}

std::vector<std::pair<std::string, long long int> > Timer::GetTimes()
{
  std::vector<std::pair<std::string, long long int> > times;
  for (const auto &timer : timers_)
  {
    times.push_back(std::make_pair(timer.first, timer.second.duration.count()));
  }
  return times;
}


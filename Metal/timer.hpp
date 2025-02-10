//
// Created by toru on 2025/02/08.
//

#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <map>

class Timer
{
  private:
    struct TimerData
    {
      std::chrono::high_resolution_clock::time_point start_time;
      std::chrono::high_resolution_clock::time_point end_time;
      std::chrono::microseconds duration;
    };

  public:
    static Timer &GetInstance();

    void Start(const std::string &name);
    void Stop(const std::string &name);
    void PrintDuration(const std::string &name);
    std::chrono::microseconds GetDuration(const std::string &name);
    std::vector<std::pair<std::string, long long int> > GetTimes();

  private:
    Timer();
    ~Timer();

    std::map<std::string, TimerData> timers_;
};

#endif //TIMER_HPP

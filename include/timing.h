
#include <map>
#include <chrono>

#pragma once

struct Time {
    double value = 0;
};


#define MARK_TIME(name) name = std::chrono::high_resolution_clock::now();

#define ADD_TIME_SINCE(timer_name, mark) \
	queue.finish(); \
    time_map[#timer_name].value += \
	std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-mark).count();


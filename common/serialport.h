#ifndef SERIAL_SENDER_H
#define SERIAL_SENDER_H

#include <string>
#include <vector>
#include <iostream>
#include <cstdarg>
#include "serial/serial.h"

// --- 新增的头文件 ---
#include <thread>   // 用于 std::thread
#include <mutex>    // 用于 std::mutex
#include <condition_variable> // 用于 std::condition_variable
#include <queue>    // 用于 std::queue
#include <atomic>   // 用于 std::atomic_bool

class SerialSender
{
 public:
	SerialSender(const std::string& port, uint32_t baudrate);
	~SerialSender();

	/**
	 * @brief [线程安全] 以 printf 格式将数据放入发送队列
	 */
	void sendFormattedData(const char* format, ...);

	bool isOpen() const;

 private:
	void sendingThread(); // 负责在后台发送数据的线程函数

	serial::Serial ser;
	bool is_open;

	// --- 线程和数据队列相关的成员 ---
	std::thread sender_thread_;
	std::queue<std::string> data_queue_;
	std::mutex queue_mutex_;
	std::condition_variable condition_;
	std::atomic_bool stop_thread_; // 用于安全地停止线程
};
extern SerialSender serial1;
#endif // SERIAL_SENDER_H

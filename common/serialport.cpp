#include "serialport.h"
#include <cstdarg>
#include <cstdio>
#include <vector>
SerialSender serial1("/dev/ttyUSB0",115200);
SerialSender::SerialSender(const std::string& port, uint32_t baudrate)
	: is_open(false), stop_thread_(false)
{
	try
	{
		ser.setPort(port);
		ser.setBaudrate(baudrate);
		serial::Timeout timeout = serial::Timeout::simpleTimeout(1000);
		ser.setTimeout(timeout);
		ser.open();
		is_open = ser.isOpen();
		if (is_open)
		{
			std::cout << "串口 " << port << " 已成功打开, 波特率: " << baudrate << std::endl;
			// 启动后台发送线程
			sender_thread_ = std::thread(&SerialSender::sendingThread, this);
		}
		else
		{
			std::cerr << "错误: 无法打开串口 " << port << std::endl;
		}
	}
	catch (serial::IOException& e)
	{
		std::cerr << "错误: 打开串口时发生异常: " << e.what() << std::endl;
	}
}

SerialSender::~SerialSender()
{
	// 通知发送线程停止
	stop_thread_ = true;
	condition_.notify_one(); // 唤醒线程以便它能检查停止标志
	if (sender_thread_.joinable())
	{
		sender_thread_.join(); // 等待线程安全退出
	}

	if (ser.isOpen())
	{
		ser.close();
		std::cout << "串口已关闭。" << std::endl;
	}
}

bool SerialSender::isOpen() const
{
	return is_open;
}

// [修改后] 这个函数现在只是把数据快速放入队列，然后立即返回
void SerialSender::sendFormattedData(const char* format, ...)
{
	if (!is_open) return;

	va_list args1;
	va_start(args1, format);
	int needed = vsnprintf(NULL, 0, format, args1);
	va_end(args1);

	if (needed < 0) {
		std::cerr << "错误: 格式化字符串时出错。" << std::endl;
		return;
	}

	std::vector<char> buffer(needed + 1);
	va_list args2;
	va_start(args2, format);
	vsnprintf(buffer.data(), buffer.size(), format, args2);
	va_end(args2);

	// --- 核心修改 ---
	// 将格式化好的字符串放入队列，而不是直接发送
	{
		std::lock_guard<std::mutex> lock(queue_mutex_); // 加锁以保证线程安全
		data_queue_.push(std::string(buffer.data(), needed));
	}
	condition_.notify_one(); // 唤醒发送线程，告诉它有新数据了
}

// [新功能] 这是在后台运行的线程函数
void SerialSender::sendingThread()
{
	while (!stop_thread_)
	{
		std::string data_to_send;
		{
			// 使用 unique_lock 和条件变量，这是线程间等待/通知的标准模式
			std::unique_lock<std::mutex> lock(queue_mutex_);
			// 线程会在这里“睡眠”，直到被 notify_one() 唤醒，或者超时
			// 并且只有在队列不为空或需要停止时，才会继续执行
			condition_.wait(lock, [this] { return !data_queue_.empty() || stop_thread_; });

			if (stop_thread_) {
				return; // 如果收到停止信号，就退出线程
			}

			// 从队列中取出数据
			data_to_send = data_queue_.front();
			data_queue_.pop();
		} // 锁在这里自动释放

		// 在这个线程里执行耗时的串口写入操作
		if (!data_to_send.empty() && ser.isOpen())
		{
			try
			{
				ser.write(data_to_send);
			}
			catch (serial::IOException& e)
			{
				std::cerr << "错误(后台线程): 发送数据时发生异常: " << e.what() << std::endl;
			}
		}
	}
}

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <functional>

namespace tnn {

    class thread_pool {
    public:
        thread_pool(std::size_t threads_n = std::thread::hardware_concurrency()) : stop(false) {
            for(; threads_n; --threads_n)
                workers.emplace_back(std::bind(&thread_pool::run, this));
        }
        thread_pool(const thread_pool &) = delete;
        thread_pool &operator = (const thread_pool &) = delete;
        thread_pool(thread_pool &&) = delete;
        thread_pool &operator = (thread_pool &&) = delete;
        template<class F, class... Args>
        std::future<typename std::result_of<F(Args...)>::type> enqueue(F&& f, Args&&... args) {
            using packaged_task_t = std::packaged_task<typename std::result_of<F(Args...)>::type ()>;
            std::shared_ptr<packaged_task_t> task(new packaged_task_t(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            ));
            auto res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                tasks.emplace([task](){ (*task)(); });
            }
            condition.notify_one();
            return res;
        }
        std::size_t get_thread_num() const {
            return workers.size();
        }
        void run() {
            while(true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this]{ return stop || !tasks.empty(); });
                    if(stop && tasks.empty())
                        return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        }
        void run_once() {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this]{ return stop || !tasks.empty(); });
                if(stop && tasks.empty())
                    return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
        ~thread_pool() {
            this->stop = true;
            this->condition.notify_all();
            for(std::thread &worker: this->workers)
                worker.join();
        }
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()> > tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        std::atomic_bool stop;
    };

}

#endif

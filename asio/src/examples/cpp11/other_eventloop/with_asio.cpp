#include "asio.hpp"

#if __has_include("log.h")
#include "log.h"
#else
#define LOG printf
#endif

int main() {
  asio::io_context io_context_main;
  io_context_main.dispatch([] { LOG("main thread"); });

  std::thread([&] {
    LOG("other thread");
    asio::io_context io_context_other;

    asio::steady_timer asio_timer(io_context_other);
    std::function<void()> start_asio_timer;
    start_asio_timer = [&] {
      asio_timer.expires_after(asio::chrono::milliseconds(1000));
      asio_timer.async_wait([&](const asio::error_code &ec) {
        LOG("asio Timer fired!");
        start_asio_timer();
      });
    };
    start_asio_timer();

    for (;;) {
      LOG("wait_event...");
      io_context_other.wait_one_for(std::chrono::minutes(1));
      std::promise<void> promise;
      io_context_main.dispatch([&] {
        io_context_other.poll_one();
        promise.set_value();
      });
      promise.get_future().get();
    }
  }).detach();

  asio::io_context::work work(io_context_main);
  io_context_main.run();
  return 0;
}

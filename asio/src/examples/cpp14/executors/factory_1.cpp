#include <asio/execution.hpp>
#include <asio/system_executor.hpp>
#include <future>
#include <iostream>
#include <string>
#include <vector>

namespace execution = asio::execution;

//------------------------------------------------------------------------------
// A simple inline executor.

struct inline_executor
{
  template <class F>
  void execute(F f) const noexcept
  {
    f();
  }

  friend bool operator==(const inline_executor&,
      const inline_executor&) noexcept
  {
    return true;
  }

  friend bool operator!=(const inline_executor&,
      const inline_executor&) noexcept
  {
    return false;
  }
};

//------------------------------------------------------------------------------
// An executor adapter that traces when functions are called.

template <typename InnerExecutor>
struct debug_executor
{
  InnerExecutor inner_;

  explicit debug_executor(const InnerExecutor& inner)
    : inner_(inner)
  {
  }

  template <class F>
  void execute(F f) const noexcept
  {
    execution::execute(inner_,
        [f = std::move(f)]() mutable
        {
          struct tracer
          {
            tracer()
            {
              std::cerr << "debug: entry\n";
            }

            ~tracer()
            {
              std::cerr << "debug: exit\n";
            }
          } t;

          f();
        });
  }

  friend bool operator==(const debug_executor& a,
      const debug_executor& b) noexcept
  {
    return a.inner_ == b.inner_;
  }

  friend bool operator!=(const debug_executor& a,
      const debug_executor& b) noexcept
  {
    return a.inner_ != b.inner_;
  }
};

//------------------------------------------------------------------------------
// Creates an executor based on some runtime configuration options.

execution::any_executor<> make_executor(
    const std::string& type, const std::vector<std::string>& options)
{
  if (type == "inline")
  {
    return inline_executor();
  }
  if (type == "system")
  {
    return asio::require(
        asio::system_executor(),
        execution::blocking.never);
  }
  if (type == "debug")
  {
    if (!options.empty())
    {
      if (execution::any_executor<> inner_executor
          = make_executor(options[0], {options.begin() + 1, options.end()}))
      {
        return debug_executor<execution::any_executor<>>(inner_executor);
      }
    }
  }
  return {};
}

//------------------------------------------------------------------------------
// An algorithm that uses an executor to run some work.

template <typename Executor>
std::future<int> times_two(const Executor& ex, int value)
{
  std::promise<int> promise;
  std::future<int> result = promise.get_future();

  execution::execute(ex,
      [value, promise = std::move(promise)]() mutable
      {
        promise.set_value(value * 2);
      });

  return result;
}

//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: factory_1 <exec_type> [<exec_options>...]\n";
    std::cerr << "  where <exec_type> is one of:\n";
    std::cerr << "    inline\n";
    std::cerr << "    system\n";
    std::cerr << "    debug <exec_type> [<exec_options>...]\n";
    return 1;
  }

  execution::any_executor<> ex = make_executor(argv[1], {argv + 2, argv + argc});
  if (!ex)
  {
    std::cerr << "Invalid executor type or options\n";
    return 1;
  }

  std::cout << "Result is: " << times_two(ex, 21).get() << "\n";
}

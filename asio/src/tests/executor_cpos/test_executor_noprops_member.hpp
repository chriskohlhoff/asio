#ifndef TEST_EXECUTOR_HPP
#define TEST_EXECUTOR_HPP

namespace ex_test {

template <std::size_t I>
class basic_test_executor
{
public:
  basic_test_executor() noexcept
  {
  }

private:
  friend bool operator==(const basic_test_executor&, const basic_test_executor&) noexcept
  {
    return true;
  }

  friend bool operator!=(const basic_test_executor&, const basic_test_executor&) noexcept
  {
    return false;
  }

public:
  template <typename Function>
  void execute(Function&& f) const
  {
    this->do_execute(static_cast<Function&&>(f));
  }

private:
  template <typename Function>
  void do_execute(Function&& f) const
  {
  }
};

} // namespace ex_test

namespace asio {
namespace execution {

template <typename>
struct is_executor;

template <std::size_t I>
struct is_executor<ex_test::basic_test_executor<I>>
{
  static constexpr bool value = true;
};

}
}

template <std::size_t I>
using test_executor = ex_test::basic_test_executor<I>;

#endif // TEST_EXECUTOR_HPP

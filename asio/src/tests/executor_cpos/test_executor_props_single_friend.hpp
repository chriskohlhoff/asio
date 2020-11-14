#ifndef TEST_EXECUTOR_MEMBER_HPP
#define TEST_EXECUTOR_MEMBER_HPP

namespace ex_test {

template <typename T>
struct is_allocator_property : std::false_type
{
};

template <typename T>
struct is_allocator_property<asio::execution::allocator_t<T>> : std::true_type
{
};

template <std::size_t I, typename Blocking, typename Relationship, typename Allocator>
class basic_test_executor
{
public:
  basic_test_executor() noexcept
    : allocator_(Allocator())
  {
  }

  template <typename RequestedProp>
  friend auto require(const basic_test_executor& ex, RequestedProp p)
    requires(
      std::convertible_to<RequestedProp, asio::execution::blocking_t::possibly_t>
        || std::convertible_to<RequestedProp, asio::execution::blocking_t::always_t>
        || std::convertible_to<RequestedProp, asio::execution::blocking_t::never_t>
        || std::convertible_to<RequestedProp, asio::execution::relationship_t::fork_t>
        || std::convertible_to<RequestedProp, asio::execution::relationship_t::continuation_t>
        || std::convertible_to<RequestedProp, asio::execution::allocator_t<void>>
        || is_allocator_property<RequestedProp>::value
    )
  {
    if constexpr (std::convertible_to<RequestedProp, asio::execution::blocking_t::possibly_t>)
      return basic_test_executor<I, asio::execution::blocking_t::possibly_t, Relationship, Allocator>(ex.allocator_);
    else if constexpr (std::convertible_to<RequestedProp, asio::execution::blocking_t::always_t>)
      return basic_test_executor<I, asio::execution::blocking_t::always_t, Relationship, Allocator>(ex.allocator_);
    else if constexpr (std::convertible_to<RequestedProp, asio::execution::blocking_t::never_t>)
      return basic_test_executor<I, asio::execution::blocking_t::never_t, Relationship, Allocator>(ex.allocator_);
    else if constexpr (std::convertible_to<RequestedProp, asio::execution::relationship_t::fork_t>)
      return basic_test_executor<I, Blocking, asio::execution::relationship_t::fork_t, Allocator>(ex.allocator_);
    else if constexpr (std::convertible_to<RequestedProp, asio::execution::relationship_t::continuation_t>)
      return basic_test_executor<I, Blocking, asio::execution::relationship_t::continuation_t, Allocator>(ex.allocator_);
    else if constexpr (std::convertible_to<RequestedProp, asio::execution::allocator_t<void>>)
      return basic_test_executor<I, Blocking, Relationship, std::allocator<void>>();
    else if constexpr (is_allocator_property<RequestedProp>::value)
      return basic_test_executor<I, Blocking, Relationship, decltype((p.value()))>(p.value());
  }

  template <typename RequestedProp>
  friend constexpr auto query(const basic_test_executor& ex, RequestedProp) noexcept
    requires(
      std::convertible_to<RequestedProp, asio::execution::mapping_t>
        || std::convertible_to<RequestedProp, asio::execution::blocking_t>
        || std::convertible_to<RequestedProp, asio::execution::relationship_t>
        || std::convertible_to<RequestedProp, asio::execution::allocator_t<void>>
        || is_allocator_property<RequestedProp>::value
    )
  {
    if constexpr (std::convertible_to<RequestedProp, asio::execution::mapping_t>)
      return asio::execution::mapping.thread;
    if constexpr (std::convertible_to<RequestedProp, asio::execution::blocking_t::possibly_t>)
      return Blocking();
    else if constexpr (std::convertible_to<RequestedProp, asio::execution::relationship_t>)
      return Relationship();
    else if constexpr (std::convertible_to<RequestedProp, asio::execution::allocator_t<void>>)
      return ex.allocator_;
    else if constexpr (is_allocator_property<RequestedProp>::value)
      return ex.allocator_;
  }

  friend bool operator==(const basic_test_executor&, const basic_test_executor&) noexcept
  {
    return true;
  }

  friend bool operator!=(const basic_test_executor&, const basic_test_executor&) noexcept
  {
    return false;
  }

  template <typename Function>
  void execute(Function&& f) const
  {
    this->do_execute(std::forward<Function>(f), Blocking());
  }

//private:
  //template <std::size_t, typename, typename, typename> friend class basic_test_executor;

  basic_test_executor(const Allocator& a)
    : allocator_(a)
  {
  }

private:
  template <typename Function>
  void do_execute(Function&& f,
      asio::execution::blocking_t::possibly_t) const
  {
  }

  template <typename Function>
  void do_execute(Function&& f,
      asio::execution::blocking_t::always_t) const
  {
  }

  template <typename Function>
  void do_execute(Function&& f,
      asio::execution::blocking_t::never_t) const
  {
  }

  Allocator allocator_;
};

} // namespace ex_test

namespace asio {
namespace execution {

template <typename>
struct is_executor;

template <std::size_t I, typename Blocking, typename Relationship, typename Allocator>
struct is_executor<ex_test::basic_test_executor<I, Blocking, Relationship, Allocator>>
{
  static constexpr bool value = true;
};

}
}

template <std::size_t I>
using test_executor = ex_test::basic_test_executor<I,
    asio::execution::blocking_t::possibly_t,
    asio::execution::relationship_t::fork_t,
    std::allocator<void> >;

#endif // TEST_EXECUTOR_MEMBER_HPP

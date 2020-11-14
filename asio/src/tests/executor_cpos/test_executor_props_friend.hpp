#ifndef TEST_EXECUTOR_MEMBER_HPP
#define TEST_EXECUTOR_MEMBER_HPP

namespace ex_test {

template <std::size_t I, typename Blocking, typename Relationship, typename Allocator>
class basic_test_executor
{
public:
  basic_test_executor() noexcept
    : allocator_(Allocator())
  {
  }

private:
  friend basic_test_executor<I, asio::execution::blocking_t::possibly_t, Relationship, Allocator>
  require(const basic_test_executor& ex, asio::execution::blocking_t::possibly_t)
  {
    return basic_test_executor<I, asio::execution::blocking_t::possibly_t, Relationship, Allocator>(ex.allocator_);
  }

  friend basic_test_executor<I, asio::execution::blocking_t::always_t, Relationship, Allocator>
  require(const basic_test_executor& ex, asio::execution::blocking_t::always_t)
  {
    return basic_test_executor<I, asio::execution::blocking_t::always_t, Relationship, Allocator>(ex.allocator_);
  }

  friend basic_test_executor<I, asio::execution::blocking_t::never_t, Relationship, Allocator>
  require(const basic_test_executor& ex, asio::execution::blocking_t::never_t)
  {
    return basic_test_executor<I, asio::execution::blocking_t::never_t, Relationship, Allocator>(ex.allocator_);
  }

  friend basic_test_executor<I, Blocking, asio::execution::relationship_t::continuation_t, Allocator>
  require(const basic_test_executor& ex, asio::execution::relationship_t::continuation_t)
  {
    return basic_test_executor<I, Blocking, asio::execution::relationship_t::continuation_t, Allocator>(ex.allocator_);
  }

  friend basic_test_executor<I, Blocking, asio::execution::relationship_t::fork_t, Allocator>
  require(const basic_test_executor& ex, asio::execution::relationship_t::fork_t)
  {
    return basic_test_executor<I, Blocking, asio::execution::relationship_t::fork_t, Allocator>(ex.allocator_);
  }

  template <typename OtherAllocator>
  friend basic_test_executor<I, Blocking, Relationship, OtherAllocator>
  require(const basic_test_executor& ex, asio::execution::allocator_t<OtherAllocator> a)
  {
    return basic_test_executor<I, Blocking, Relationship, OtherAllocator>(a.value());
  }

  friend basic_test_executor<I, Blocking, Relationship, std::allocator<void> >
  require(const basic_test_executor& ex, asio::execution::allocator_t<void>)
  {
    return basic_test_executor<I, Blocking, Relationship, std::allocator<void> >();
  }

public:
  static constexpr asio::execution::mapping_t query(asio::execution::mapping_t) noexcept
  {
    return asio::execution::mapping.thread;
  }

private:
  friend constexpr asio::execution::blocking_t query(const basic_test_executor&, asio::execution::blocking_t) noexcept
  {
    return Blocking();
  }

  friend constexpr asio::execution::relationship_t query(const basic_test_executor&, asio::execution::relationship_t) noexcept
  {
    return Relationship();
  }

  template <typename OtherAllocator>
  friend constexpr Allocator query(const basic_test_executor& ex, asio::execution::allocator_t<OtherAllocator>) noexcept
  {
    return ex.allocator_;
  }

  friend constexpr Allocator query(const basic_test_executor& ex, asio::execution::allocator_t<void>) noexcept
  {
    return ex.allocator_;
  }

  friend std::size_t query(const basic_test_executor& ex, asio::execution::occupancy_t) noexcept
  {
    return 1;
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
  friend void execute(const basic_test_executor& ex, Function&& f)
  {
    ex.do_execute(std::forward<Function>(f), Blocking());
  }

private:
  template <std::size_t, typename, typename, typename> friend class basic_test_executor;

  basic_test_executor(const Allocator& a)
    : allocator_(a)
  {
  }

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

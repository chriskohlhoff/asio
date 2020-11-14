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

  basic_test_executor<I, asio::execution::blocking_t::possibly_t, Relationship, Allocator>
  require(asio::execution::blocking_t::possibly_t) const
  {
    return basic_test_executor<I, asio::execution::blocking_t::possibly_t, Relationship, Allocator>(allocator_);
  }

  basic_test_executor<I, asio::execution::blocking_t::always_t, Relationship, Allocator>
  require(asio::execution::blocking_t::always_t) const
  {
    return basic_test_executor<I, asio::execution::blocking_t::always_t, Relationship, Allocator>(allocator_);
  }

  basic_test_executor<I, asio::execution::blocking_t::never_t, Relationship, Allocator>
  require(asio::execution::blocking_t::never_t) const
  {
    return basic_test_executor<I, asio::execution::blocking_t::never_t, Relationship, Allocator>(allocator_);
  }

  basic_test_executor<I, Blocking, asio::execution::relationship_t::continuation_t, Allocator>
  require(asio::execution::relationship_t::continuation_t) const
  {
    return basic_test_executor<I, Blocking, asio::execution::relationship_t::continuation_t, Allocator>(allocator_);
  }

  basic_test_executor<I, Blocking, asio::execution::relationship_t::fork_t, Allocator>
  require(asio::execution::relationship_t::fork_t) const
  {
    return basic_test_executor<I, Blocking, asio::execution::relationship_t::fork_t, Allocator>(allocator_);
  }

  template <typename OtherAllocator>
  basic_test_executor<I, Blocking, Relationship, OtherAllocator>
  require(asio::execution::allocator_t<OtherAllocator> a) const
  {
    return basic_test_executor<I, Blocking, Relationship, OtherAllocator>(a.value());
  }

  basic_test_executor<I, Blocking, Relationship, std::allocator<void> >
  require(asio::execution::allocator_t<void>) const
  {
    return basic_test_executor<I, Blocking, Relationship, std::allocator<void> >();
  }

  static constexpr asio::execution::mapping_t query(asio::execution::mapping_t) noexcept
  {
    return asio::execution::mapping.thread;
  }

  static constexpr asio::execution::blocking_t query(asio::execution::blocking_t) noexcept
  {
    return Blocking();
  }

  static constexpr asio::execution::relationship_t query(asio::execution::relationship_t) noexcept
  {
    return Relationship();
  }

  template <typename OtherAllocator>
  constexpr Allocator query(asio::execution::allocator_t<OtherAllocator>) const noexcept
  {
    return allocator_;
  }

  constexpr Allocator query(asio::execution::allocator_t<void>) const noexcept
  {
    return allocator_;
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

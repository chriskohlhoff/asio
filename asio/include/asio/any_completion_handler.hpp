//
// any_completion_handler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2022 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ANY_COMPLETION_HANDLER_HPP
#define ASIO_ANY_COMPLETION_HANDLER_HPP

#include "asio/detail/config.hpp"
#include <functional>
#include <memory>
#include <utility>
#include "asio/any_completion_executor.hpp"
#include "asio/associated_allocator.hpp"
#include "asio/associated_cancellation_slot.hpp"
#include "asio/associated_executor.hpp"
#include "asio/cancellation_state.hpp"
#include "asio/recycling_allocator.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class any_completion_handler_impl_base
{
public:
  template <typename S>
  explicit any_completion_handler_impl_base(S&& slot)
    : cancel_state_(std::forward<S>(slot), enable_total_cancellation())
  {
  }

  cancellation_slot get_cancellation_slot() const noexcept
  {
    return cancel_state_.slot();
  }

private:
  cancellation_state cancel_state_;
};

template <typename Handler>
class any_completion_handler_impl :
  public any_completion_handler_impl_base
{
public:
  template <typename S, typename H>
  any_completion_handler_impl(S&& slot, H&& h)
    : any_completion_handler_impl_base(std::forward<S>(slot)),
      handler_(std::forward<H>(h))
  {
  }

  struct uninit_deleter
  {
    typename std::allocator_traits<
      associated_allocator_t<Handler,
        asio::recycling_allocator<void>>>::template
          rebind_alloc<any_completion_handler_impl> alloc;

    void operator()(any_completion_handler_impl* ptr)
    {
      std::allocator_traits<decltype(alloc)>::deallocate(alloc, ptr, 1);
    }
  };

  struct deleter
  {
    typename std::allocator_traits<
      associated_allocator_t<Handler,
        asio::recycling_allocator<void>>>::template
          rebind_alloc<any_completion_handler_impl> alloc;

    void operator()(any_completion_handler_impl* ptr)
    {
      std::allocator_traits<decltype(alloc)>::destroy(alloc, ptr);
      std::allocator_traits<decltype(alloc)>::deallocate(alloc, ptr, 1);
    }
  };

  template <typename S, typename H>
  static any_completion_handler_impl* create(S&& slot, H&& h)
  {
    uninit_deleter d{(get_associated_allocator)(h, asio::recycling_allocator<void>())};
    std::unique_ptr<any_completion_handler_impl, uninit_deleter> uninit_ptr(
        std::allocator_traits<decltype(d.alloc)>::allocate(d.alloc, 1), d);

    any_completion_handler_impl* ptr =
      new (uninit_ptr.get()) any_completion_handler_impl(
        std::forward<S>(slot), std::forward<H>(h));

    uninit_ptr.release();
    return ptr;
  }

  void destroy()
  {
    deleter d{(get_associated_allocator)(handler_, asio::recycling_allocator<void>())};
    d(this);
  }

  any_completion_executor executor(const any_completion_executor& candidate) const
  {
    return (get_associated_executor)(handler_, candidate);
  }

  template <typename... Args>
  void call(Args&&... args)
  {
    deleter d{(get_associated_allocator)(handler_, asio::recycling_allocator<void>())};
    std::unique_ptr<any_completion_handler_impl, deleter> ptr(this, d);
    Handler handler(std::move(handler_));
    ptr.reset();
    std::move(handler)(std::forward<Args>(args)...);
  }

private:
  Handler handler_;
};

template <typename Signature>
class any_completion_handler_call_fn;

template <typename R, typename... Args>
class any_completion_handler_call_fn<R(Args...)>
{
public:
  using type = void(*)(any_completion_handler_impl_base*, Args...);

  constexpr any_completion_handler_call_fn(type fn)
    : call_fn_(fn)
  {
  }

  void call(any_completion_handler_impl_base* impl, Args... args) const
  {
    call_fn_(impl, std::move(args)...);
  }

  template <typename Handler>
  static void impl(any_completion_handler_impl_base* impl, Args... args)
  {
    static_cast<any_completion_handler_impl<Handler>*>(impl)->call(std::move(args)...);
  }

private:
  type call_fn_;
};

template <typename... Signatures>
class any_completion_handler_call_fns;

template <typename Signature>
class any_completion_handler_call_fns<Signature> :
  public any_completion_handler_call_fn<Signature>
{
public:
  using any_completion_handler_call_fn<Signature>::any_completion_handler_call_fn;
  using any_completion_handler_call_fn<Signature>::call;
};

template <typename Signature, typename... Signatures>
class any_completion_handler_call_fns<Signature, Signatures...> :
  public any_completion_handler_call_fn<Signature>,
  public any_completion_handler_call_fns<Signatures...>
{
public:
  template <typename CallFn, typename... CallFns>
  constexpr any_completion_handler_call_fns(CallFn fn, CallFns... fns)
    : any_completion_handler_call_fn<Signature>(fn),
      any_completion_handler_call_fns<Signatures...>(fns...)
  {
  }

  using any_completion_handler_call_fn<Signature>::call;
  using any_completion_handler_call_fns<Signatures...>::call;
};

class any_completion_handler_destroy_fn
{
public:
  using type = void(*)(any_completion_handler_impl_base*);

  constexpr any_completion_handler_destroy_fn(type fn)
    : destroy_fn_(fn)
  {
  }

  void destroy(any_completion_handler_impl_base* impl) const
  {
    destroy_fn_(impl);
  }

  template <typename Handler>
  static void impl(any_completion_handler_impl_base* impl)
  {
    static_cast<any_completion_handler_impl<Handler>*>(impl)->destroy();
  }

private:
  type destroy_fn_;
};

class any_completion_handler_executor_fn
{
public:
  using type = any_completion_executor(*)(any_completion_handler_impl_base*, const any_completion_executor&);

  constexpr any_completion_handler_executor_fn(type fn)
    : executor_fn_(fn)
  {
  }

  any_completion_executor executor(any_completion_handler_impl_base* impl, const any_completion_executor& candidate) const
  {
    return executor_fn_(impl, candidate);
  }

  template <typename Handler>
  static any_completion_executor impl(any_completion_handler_impl_base* impl, const any_completion_executor& candidate)
  {
    return static_cast<any_completion_handler_impl<Handler>*>(impl)->executor(candidate);
  }

private:
  type executor_fn_;
};

template <typename... Signatures>
class any_completion_handler_fn_table
  : private any_completion_handler_destroy_fn,
    private any_completion_handler_executor_fn,
    private any_completion_handler_call_fns<Signatures...>
{
public:
  template <typename... CallFns>
  constexpr any_completion_handler_fn_table(
      any_completion_handler_destroy_fn::type destroy_fn,
      any_completion_handler_executor_fn::type executor_fn,
      CallFns... call_fns)
    : any_completion_handler_destroy_fn(destroy_fn),
      any_completion_handler_executor_fn(executor_fn),
      any_completion_handler_call_fns<Signatures...>(call_fns...)
  {
  }

  using any_completion_handler_destroy_fn::destroy;
  using any_completion_handler_executor_fn::executor;
  using any_completion_handler_call_fns<Signatures...>::call;
};

template <typename Handler, typename... Signatures>
struct any_completion_handler_fn_table_instance
{
  static constexpr any_completion_handler_fn_table<Signatures...>
    value = any_completion_handler_fn_table<Signatures...>(
        &any_completion_handler_destroy_fn::impl<Handler>,
        &any_completion_handler_executor_fn::impl<Handler>,
        &any_completion_handler_call_fn<Signatures>::template impl<Handler>...);
};

template <typename Handler, typename... Signatures>
constexpr any_completion_handler_fn_table<Signatures...>
any_completion_handler_fn_table_instance<Handler, Signatures...>::value;

} // namespace detail

template <typename... Signatures>
class any_completion_handler
{
private:
  template <typename, typename> friend struct associated_executor;
  const detail::any_completion_handler_fn_table<Signatures...>* fn_table_;
  detail::any_completion_handler_impl_base* impl_;

public:
  using cancellation_slot_type = cancellation_slot;

  constexpr any_completion_handler()
    : fn_table_(nullptr),
      impl_(nullptr)
  {
  }

  constexpr any_completion_handler(nullptr_t)
    : fn_table_(nullptr),
      impl_(nullptr)
  {
  }

  template <typename H, typename Handler = typename decay<H>::type>
  any_completion_handler(H&& h)
    : fn_table_(&detail::any_completion_handler_fn_table_instance<Handler, Signatures...>::value),
      impl_(detail::any_completion_handler_impl<Handler>::create(
            (get_associated_cancellation_slot)(h), std::forward<H>(h)))
  {
  }

  any_completion_handler(any_completion_handler&& other) noexcept
    : fn_table_(other.fn_table_),
      impl_(other.impl_)
  {
    other.fn_table_ = nullptr;
    other.impl_ = nullptr;
  }

  any_completion_handler& operator=(any_completion_handler&& other) noexcept
  {
    any_completion_handler(other).swap(*this);
    return *this;
  }

  any_completion_handler& operator=(nullptr_t) noexcept
  {
    any_completion_handler().swap(*this);
    return *this;
  }

  ~any_completion_handler()
  {
    if (impl_)
      fn_table_->destroy(impl_);
  }

  constexpr explicit operator bool() const noexcept
  {
    return impl_ != nullptr;
  }

  constexpr bool operator!() const noexcept
  {
    return impl_ == nullptr;
  }

  void swap(any_completion_handler& other) noexcept
  {
    std::swap(fn_table_, other.fn_table_);
    std::swap(impl_, other.impl_);
  }

  cancellation_slot_type get_cancellation_slot() const noexcept
  {
    return impl_->get_cancellation_slot();
  }

  template <typename... Args>
  auto operator()(Args&&... args)
    -> decltype(fn_table_->call(impl_, std::forward<Args>(args)...))
  {
    if (detail::any_completion_handler_impl_base* impl = impl_)
    {
      impl_ = nullptr;
      return fn_table_->call(impl, std::forward<Args>(args)...);
    }
    std::bad_function_call ex;
    asio::detail::throw_exception(ex);
  }

  friend constexpr bool operator==(const any_completion_handler& a, nullptr_t) noexcept
  {
    return a.impl_ == nullptr;
  }

  friend constexpr bool operator==(nullptr_t, const any_completion_handler& b) noexcept
  {
    return nullptr == b.impl_;
  }

  friend constexpr bool operator!=(const any_completion_handler& a, nullptr_t) noexcept
  {
    return a.impl_ != nullptr;
  }

  friend constexpr bool operator!=(nullptr_t, const any_completion_handler& b) noexcept
  {
    return nullptr != b.impl_;
  }
};

template <typename... Signatures, typename Candidate>
struct associated_executor<any_completion_handler<Signatures...>, Candidate>
{
  using type = any_completion_executor;

  static type get(const any_completion_handler<Signatures...>& handler, const type& candidate = Candidate()) noexcept
  {
    return handler.fn_table_->executor(handler.impl_, candidate);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ANY_COMPLETION_HANDLER_HPP

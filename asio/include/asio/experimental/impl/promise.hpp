//
// experimental/promise.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef ASIO_EXPERIMENTAL_IMPL_PROMISE_HPP
#define ASIO_EXPERIMENTAL_IMPL_PROMISE_HPP

#include <asio/cancellation_signal.hpp>
#include <asio/experimental/impl/completion_handler_erasure.hpp>

#include <tuple>
#include <optional>

namespace asio {
namespace experimental {

template<typename Signature = void(), typename Executor = any_io_executor>
struct promise;

namespace detail {

template<typename Signature, typename Executor>
struct promise_impl;

template<typename ... Ts, typename Executor>
struct promise_impl<void(Ts...), Executor>
{
    using result_type = std::tuple<Ts...>;
    std::optional<result_type> result;
    bool done{false};
    detail::completion_handler_erasure<void(Ts...), Executor> completion;
    cancellation_signal cancel;

    promise_impl(Executor executor = {}) : executor(std::move(executor)) {}
    Executor executor;
};

template<typename Signature = void(), typename Executor = any_io_executor>
struct promise_handler;



template<typename Signature, typename Executor>
struct promise_handler;


template<typename ... Ts, typename Executor>
struct promise_handler<void(Ts...), Executor>
{
    using promise_type = promise<void(Ts...), Executor>;

    promise_handler(Executor executor) // get_associated_allocator(exec)
            : impl_{std::allocate_shared<promise_impl<void(Ts...), Executor>>(get_associated_allocator(executor))}
    {
        impl_->executor = std::move(executor);
    }

    std::shared_ptr<promise_impl<void(Ts...), Executor>> impl_;

    using cancellation_slot_type = cancellation_slot;
    cancellation_slot_type get_cancellation_slot() const
    {
        return impl_->cancel.slot();
    }

    auto make_promise() -> promise<void(Ts...), Executor>
    {
        return {impl_};
    }
    void operator()(std::remove_reference_t<Ts> ... ts)
    {
        assert(impl_);
        impl_->result.emplace(std::move(ts)...);
        impl_->done = true;
        if (auto f = std::exchange(impl_->completion, nullptr); f != nullptr)
            std::apply(std::move(f), std::move(*impl_->result));
    }
};




}
}
}
#endif //ASIO_EXPERIMENTAL_IMPL_PROMISE_HPP

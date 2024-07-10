//
// experimental/awaitable_specific_ptr.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2024 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_AWAITABLE_SPECIFIC_PTR_HPP
#define ASIO_EXPERIMENTAL_AWAITABLE_SPECIFIC_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <mutex>
#include <unordered_set>
#include <unordered_map>

#if defined(ASIO_ENABLE_HANDLER_TRACKING)
# if defined(ASIO_HAS_SOURCE_LOCATION)
#  include "asio/detail/source_location.hpp"
# endif // defined(ASIO_HAS_SOURCE_LOCATION)
#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail {
    template<typename> class awaitable_frame_base;
}

namespace experimental {
namespace detail {

class awaitable_specific_ptr_base;

class awaitable_storage_class {
    std::unordered_map<asio::experimental::detail::awaitable_specific_ptr_base *, void *> storage_;

public:
    ~awaitable_storage_class();

    void *get(const asio::experimental::detail::awaitable_specific_ptr_base *ptr) const;
    void set(asio::experimental::detail::awaitable_specific_ptr_base *ptr, void *value);
    void *release(asio::experimental::detail::awaitable_specific_ptr_base *ptr);
};

class awaitable_specific_ptr_base {
    friend class awaitable_storage_class;

protected:
    std::mutex stores_lock_;
    std::unordered_set<awaitable_storage_class *> stores_;

    virtual void cleanup(void *value) = 0;

    awaitable_specific_ptr_base() {}
public:
    virtual ~awaitable_specific_ptr_base() {}

    template<typename T>
    struct get_t {
        const awaitable_specific_ptr_base *base;
    };
    template<typename T>
    struct release_t {
        awaitable_specific_ptr_base *base;
    };
    template<typename T>
    struct reset_t {
        awaitable_specific_ptr_base *base;
        T *value;
    };

};

inline awaitable_storage_class::~awaitable_storage_class() {
  for (auto &storage : storage_) {
    {
      std::scoped_lock sl(storage.first->stores_lock_);
      storage.first->stores_.erase(this);
    }
    storage.first->cleanup(storage.second);
  }
}

inline void *awaitable_storage_class::get(const asio::experimental::detail::awaitable_specific_ptr_base *ptr) const {
  auto itr = storage_.find(const_cast<asio::experimental::detail::awaitable_specific_ptr_base *>(ptr));
  if (itr != storage_.end()) {
    return itr->second;
  }
  return nullptr;
}

inline void awaitable_storage_class::set(asio::experimental::detail::awaitable_specific_ptr_base *ptr, void *value) {
  if (!value) {
    auto itr = storage_.find(ptr);
    if (itr != storage_.end()) {
      {
        std::scoped_lock sl(itr->first->stores_lock_);
        itr->first->stores_.erase(this);
      }
      itr->first->cleanup(itr->second);
      storage_.erase(itr);
    }
    return;
  }

  auto [itr, created] = storage_.try_emplace(ptr, value);
  if (created) {
    std::scoped_lock sl(itr->first->stores_lock_);
    itr->first->stores_.insert(this);
  } else {
    itr->first->cleanup(itr->second);
    itr->second = value;
  }
}

inline void *awaitable_storage_class::release(asio::experimental::detail::awaitable_specific_ptr_base *ptr) {
  auto itr = storage_.find(ptr);
  if (itr != storage_.end()) {
    void *rv = itr->second;
    storage_.erase(itr);
    return rv;
  }
  return nullptr;
}

} // namespace detail

template <typename T>
class awaitable_specific_ptr : public detail::awaitable_specific_ptr_base
{
    void (*cleanup_function_)(T*);

    void cleanup(void *value) {
        cleanup_function_((T*) value);
    }
public:
    awaitable_specific_ptr() : cleanup_function_([](T *in) { delete in; }) {}
    explicit awaitable_specific_ptr(void (*cleanup_function)(T*)) : cleanup_function_(cleanup_function) {}
    ~awaitable_specific_ptr() {
      for (const auto store : stores_) {
        cleanup(store->release(this));
      }
    }

    // co_await get()
    get_t<T> get() const {
      return get_t<T>{this};
    }

    T* operator->() const { return get(); }
    T& operator*() const { return *get(); }

    // co_await release()
    release_t<T> release() {
      return release_t<T>{this};
    }

    // co_await reset()
    reset_t<T> reset(T* new_value=0) {
      return reset_t<T>{this, new_value};
    }
};

} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif //ASIO_EXPERIMENTAL_AWAITABLE_SPECIFIC_PTR_HPP

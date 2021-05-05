//
// detail/aligned_alloc.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Maarten de Vries (maarten at de-vri dot es)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_ALIGNED_ALLOC_HPP
#define ASIO_DETAIL_ALIGNED_ALLOC_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_ALIGNED_NEW) \
  && defined(ASIO_HAS_BOOST_ALIGN) \
  && defined(ASIO_HAS_ALIGNOF)
# include <boost/align/aligned_alloc.hpp>
#else
# include <new>
#endif

namespace asio {
namespace detail {

inline void* aligned_alloc(std::size_t align, std::size_t size) {
#if defined(ASIO_HAS_ALIGNED_NEW) && defined(ASIO_HAS_ALIGNOF)
  return ::operator new(size, std::align_val_t(align));
#elif defined(ASIO_HAS_BOOST_ALIGN) && defined(ASIO_HAS_ALIGNOF)
  return boost::alignment::aligned_alloc(align, size);
#else
  (void) align;
  return ::operator new(size);
#endif
}

inline void aligned_free(void* ptr) {
#if !defined(ASIO_HAS_ALIGNED_NEW) \
  && defined(ASIO_HAS_BOOST_ALIGN) \
  && defined(ASIO_HAS_ALIGNOF)
  boost::alignment::aligned_free(ptr);
#else
  ::operator delete(ptr);
#endif
}

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_ALIGNED_ALLOC_HPP

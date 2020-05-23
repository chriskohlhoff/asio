//
// detail/apple_nw_buffer_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_APPLE_NW_BUFFER_HELPERS_HPP
#define ASIO_DETAIL_APPLE_NW_BUFFER_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/buffer.hpp"
#include "asio/detail/apple_nw_ptr.hpp"
#include <Network/Network.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Iterator>
apple_nw_ptr<dispatch_data_t> apple_nw_buffers_to_dispatch_data(
    Iterator begin, Iterator end)
{
  apple_nw_ptr<dispatch_data_t> data(dispatch_data_empty);
  for (Iterator iter = begin; iter != end; ++iter)
  {
    asio::const_buffer buffer(*iter);
    data.reset(dispatch_data_create_concat(data,
          dispatch_data_create(buffer.data(), buffer.size(), 0, ^{})));
  }
  return data;
}

template <typename ConstBufferSequence>
inline apple_nw_ptr<dispatch_data_t> apple_nw_buffers_to_dispatch_data(
    const ConstBufferSequence& buffers)
{
  return apple_nw_buffers_to_dispatch_data(
      asio::buffer_sequence_begin(buffers),
      asio::buffer_sequence_end(buffers));
}

template <typename Iterator>
inline std::size_t apple_nw_buffers_from_dispatch_data(
    Iterator begin, Iterator end, const apple_nw_ptr<dispatch_data_t>& data)
{
  std::size_t result = 0;
  if (begin != end && !!data)
  {
    Iterator iter = begin;
    asio::mutable_buffer current_buffer(*iter);
    __block asio::mutable_buffer& current_buffer_ref = current_buffer;
    __block Iterator& iter_ref = iter;
    __block std::size_t& result_ref = result;
    dispatch_data_apply(data,
        ^(dispatch_data_t, std::size_t,
          const void* buffer, std::size_t size)
        {
          asio::const_buffer source_buffer(buffer, size);
          while (source_buffer.size() > 0)
          {
            while (current_buffer_ref.size() == 0)
            {
              ++iter_ref;
              if (iter_ref == end)
                return false;
              current_buffer_ref = asio::mutable_buffer(*iter_ref);
            }
            std::size_t n = asio::buffer_copy(
                current_buffer_ref, source_buffer);
            current_buffer_ref += n;
            source_buffer += n;
            result_ref += n;
          }
          return true;
        });
  }
  return result;
}

template <typename ConstBufferSequence>
inline std::size_t apple_nw_buffers_from_dispatch_data(
    ConstBufferSequence& buffers, const apple_nw_ptr<dispatch_data_t>& data)
{
  return apple_nw_buffers_from_dispatch_data(
      asio::buffer_sequence_begin(buffers),
      asio::buffer_sequence_end(buffers), data);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_APPLE_NW_BUFFER_HELPERS_HPP

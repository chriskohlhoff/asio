//
// detail/impl/buffer_sequence_adapter.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_BUFFER_SEQUENCE_ADAPTER_IPP
#define ASIO_DETAIL_IMPL_BUFFER_SEQUENCE_ADAPTER_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include <robuffer.h>
#include <winrt/Windows.Storage.Streams.h>
#include <wrl/implements.h>
#include "asio/detail/buffer_sequence_adapter.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class winrt_buffer_impl : public winrt::implements < winrt_buffer_impl,
		winrt::Windows::Storage::Streams::IBuffer, Windows::Storage::Streams::IBufferByteAccess>
{
public:
  explicit winrt_buffer_impl(const asio::const_buffer& b)
  {
    bytes_ = const_cast<byte*>(static_cast<const byte*>(b.data()));
    length_ = b.size();
    capacity_ = b.size();
  }

  explicit winrt_buffer_impl(const asio::mutable_buffer& b)
  {
    bytes_ = static_cast<byte*>(b.data());
    length_ = 0;
    capacity_ = b.size();
  }

  ~winrt_buffer_impl()
  {
  }

  STDMETHODIMP Buffer(byte** value) override
  {
    *value = bytes_;
    return S_OK;
  }

    uint32_t Capacity()
  {
    return capacity_;
  }

  uint32_t Length()
  {
	  return length_;
  }

  STDMETHODIMP Length(UINT32 value)
  {
	  if (value > capacity_)
		  return E_INVALIDARG;
	  length_ = value;
	  return S_OK;
  }

private:
  byte* bytes_;
  UINT32 length_;
  UINT32 capacity_;
};

void buffer_sequence_adapter_base::init_native_buffer(
    buffer_sequence_adapter_base::native_buffer_type& buf,
    const asio::mutable_buffer& buffer)
{
	buf = winrt::make<winrt_buffer_impl>(buffer).as<buffer_sequence_adapter_base::native_buffer_type>();
}

void buffer_sequence_adapter_base::init_native_buffer(
    buffer_sequence_adapter_base::native_buffer_type& buf,
    const asio::const_buffer& buffer)
{
	buf = winrt::make<winrt_buffer_impl>(buffer).as<buffer_sequence_adapter_base::native_buffer_type>();
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_IMPL_BUFFER_SEQUENCE_ADAPTER_IPP

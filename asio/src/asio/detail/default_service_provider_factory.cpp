//
// default_service_provider_factory.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#include "asio/detail/default_service_provider_factory.hpp"
#include "asio/detail/shared_thread_demuxer_provider.hpp"
#include "asio/detail/select_provider.hpp"
#include "asio/detail/timer_queue_provider.hpp"
#include "asio/detail/win_iocp_provider.hpp"

namespace asio {
namespace detail {

service_provider*
default_service_provider_factory::
create_service_provider(
    demuxer& owning_demuxer,
    const service_type_id& service_type)
{
#if 0 // defined(_WIN32)
  // TODO - Add this back in when IO completion port provider is finished.
  if (service_type == demuxer_service::id
      || service_type == dgram_socket_service::id
      || service_type == stream_socket_service::id)
    return new win_iocp_provider;
#endif

  if (service_type == demuxer_service::id)
    return new shared_thread_demuxer_provider;

  if (service_type == dgram_socket_service::id
      || service_type == socket_acceptor_service::id
      || service_type == socket_connector_service::id
      || service_type == stream_socket_service::id)
    return new select_provider(owning_demuxer);

  if (service_type == timer_queue_service::id)
    return new timer_queue_provider(owning_demuxer);

  return 0;
}

} // namespace detail
} // namespace asio

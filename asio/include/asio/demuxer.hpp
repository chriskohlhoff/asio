//
// demuxer.hpp
// ~~~~~~~~~~~
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

#ifndef ASIO_DEMUXER_HPP
#define ASIO_DEMUXER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/service_factory.hpp"
#if defined(_WIN32)
# include "asio/detail/win_iocp_demuxer_service.hpp"
#else
# include "asio/detail/select_reactor.hpp"
# include "asio/detail/task_demuxer_service.hpp"
#endif

namespace asio {

/// Typedef for typical usage of demuxer.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_demuxer
  <
    implementation_defined
  > demuxer;
#elif defined(_WIN32)
typedef basic_demuxer
  <
    detail::win_iocp_demuxer_service
  > demuxer;
#else
typedef basic_demuxer
  <
    detail::task_demuxer_service
      <
        detail::select_reactor
      >
  > demuxer;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEMUXER_HPP

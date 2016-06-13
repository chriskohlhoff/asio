//
// impl/src.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_SRC_HPP
#define ASIO_IMPL_SRC_HPP

#define ASIO_SOURCE

#include "../detail/config.hpp"

#if defined(ASIO_HEADER_ONLY)
# error Do not compile Asio library source with ASIO_HEADER_ONLY defined
#endif

#include "../impl/error.ipp"
#include "../impl/error_code.ipp"
#include "../impl/execution_context.ipp"
#include "../impl/executor.ipp"
#include "../impl/handler_alloc_hook.ipp"
#include "../impl/io_context.ipp"
#include "../impl/serial_port_base.ipp"
#include "../impl/system_executor.ipp"
#include "../impl/thread_pool.ipp"
#include "../detail/impl/buffer_sequence_adapter.ipp"
#include "../detail/impl/descriptor_ops.ipp"
#include "../detail/impl/dev_poll_reactor.ipp"
#include "../detail/impl/epoll_reactor.ipp"
#include "../detail/impl/eventfd_select_interrupter.ipp"
#include "../detail/impl/handler_tracking.ipp"
#include "../detail/impl/kqueue_reactor.ipp"
#include "../detail/impl/pipe_select_interrupter.ipp"
#include "../detail/impl/posix_event.ipp"
#include "../detail/impl/posix_mutex.ipp"
#include "../detail/impl/posix_thread.ipp"
#include "../detail/impl/posix_tss_ptr.ipp"
#include "../detail/impl/reactive_descriptor_service.ipp"
#include "../detail/impl/reactive_serial_port_service.ipp"
#include "../detail/impl/reactive_socket_service_base.ipp"
#include "../detail/impl/resolver_service_base.ipp"
#include "../detail/impl/scheduler.ipp"
#include "../detail/impl/select_reactor.ipp"
#include "../detail/impl/service_registry.ipp"
#include "../detail/impl/signal_set_service.ipp"
#include "../detail/impl/socket_ops.ipp"
#include "../detail/impl/socket_select_interrupter.ipp"
#include "../detail/impl/strand_executor_service.ipp"
#include "../detail/impl/strand_service.ipp"
#include "../detail/impl/throw_error.ipp"
#include "../detail/impl/timer_queue_ptime.ipp"
#include "../detail/impl/timer_queue_set.ipp"
#include "../detail/impl/win_iocp_handle_service.ipp"
#include "../detail/impl/win_iocp_io_context.ipp"
#include "../detail/impl/win_iocp_serial_port_service.ipp"
#include "../detail/impl/win_iocp_socket_service_base.ipp"
#include "../detail/impl/win_event.ipp"
#include "../detail/impl/win_mutex.ipp"
#include "../detail/impl/win_object_handle_service.ipp"
#include "../detail/impl/win_static_mutex.ipp"
#include "../detail/impl/win_thread.ipp"
#include "../detail/impl/win_tss_ptr.ipp"
#include "../detail/impl/winrt_ssocket_service_base.ipp"
#include "../detail/impl/winrt_timer_scheduler.ipp"
#include "../detail/impl/winsock_init.ipp"
#include "../generic/detail/impl/endpoint.ipp"
#include "../ip/impl/address.ipp"
#include "../ip/impl/address_v4.ipp"
#include "../ip/impl/address_v6.ipp"
#include "../ip/impl/host_name.ipp"
#include "../ip/impl/network_v4.ipp"
#include "../ip/impl/network_v6.ipp"
#include "../ip/detail/impl/endpoint.ipp"
#include "../local/detail/impl/endpoint.ipp"

#endif // ASIO_IMPL_SRC_HPP

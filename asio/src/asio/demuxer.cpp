//
// demuxer.cpp
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

#include "asio/demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/detail/service_registry.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {

#if defined(_WIN32)
namespace
{
  class winsock_init
  {
  public:
    winsock_init()
    {
      WSADATA wsa_data;
      ::WSAStartup(MAKEWORD(2, 0), &wsa_data);
    }

    ~winsock_init()
    {
      ::WSACleanup();
    }
  };

  winsock_init the_winsock_initialiser;
}
#endif

demuxer::
demuxer()
  : service_registry_(new detail::service_registry(*this,
        service_provider_factory::default_factory())),
    service_(dynamic_cast<demuxer_service&>(
          service_registry_->get_service(demuxer_service::id)))
{
}

demuxer::
demuxer(
    service_provider_factory& sp_factory)
  : service_registry_(new detail::service_registry(*this, sp_factory)),
    service_(dynamic_cast<demuxer_service&>(
          service_registry_->get_service(demuxer_service::id)))
{
}

demuxer::
~demuxer()
{
  delete service_registry_;
}

void
demuxer::
run()
{
  service_.run();
}

void
demuxer::
interrupt()
{
  service_.interrupt();
}

void
demuxer::
reset()
{
  service_.reset();
}

void
demuxer::
add_task(
    demuxer_task& task,
    void* arg)
{
  service_.add_task(task, arg);
}

void
demuxer::
operation_started()
{
  service_.operation_started();
}

void
demuxer::
operation_completed(
    const completion_handler& handler,
    completion_context& context,
    bool allow_nested_delivery)
{
  service_.operation_completed(handler, context, allow_nested_delivery);
}

void
demuxer::
operation_immediate(
    const completion_handler& handler,
    completion_context& context,
    bool allow_nested_delivery)
{
  service_.operation_immediate(handler, context, allow_nested_delivery);
}

service&
demuxer::
get_service(
    const service_type_id& type)
{
  return service_registry_->get_service(type);
}

} // namespace asio

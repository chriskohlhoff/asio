//
// demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DEMUXER_SERVICE_HPP
#define ASIO_DEMUXER_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/demuxer.hpp"
#include "asio/service.hpp"
#include "asio/service_type_id.hpp"

namespace asio {

/// The demuxer_service class is a base class for service implementations that
/// provide the functionality required by the demuxer class.
class demuxer_service
  : public virtual service
{
public:
  typedef demuxer::completion_handler completion_handler;

  /// The service type id.
  static const service_type_id id;

  /// Run the demuxer's event processing loop.
  virtual void run() = 0;

  /// Interrupt the demuxer's event processing loop.
  virtual void interrupt() = 0;

  /// Reset the demuxer in preparation for a subsequent run invocation.
  virtual void reset() = 0;

  /// Add a task to the demuxer.
  virtual void add_task(demuxer_task& task, void* arg) = 0;

  // Notify the demuxer that an operation has started.
  virtual void operation_started() = 0;

  /// Notify the demuxer that an operation has completed.
  virtual void operation_completed(const completion_handler& handler,
      completion_context& context, bool allow_nested_delivery) = 0;

  /// Notify the demuxer of an operation that started and finished immediately.
  virtual void operation_immediate(const completion_handler& handler,
      completion_context& context, bool allow_nested_delivery) = 0;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEMUXER_SERVICE_HPP

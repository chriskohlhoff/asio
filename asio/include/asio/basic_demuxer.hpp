//
// basic_demuxer.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_BASIC_DEMUXER_HPP
#define ASIO_BASIC_DEMUXER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/null_completion_context.hpp"
#include "asio/service_factory.hpp"
#include "asio/detail/service_registry.hpp"
#include "asio/detail/signal_init.hpp"
#include "asio/detail/winsock_init.hpp"

namespace asio {

/// The basic_demuxer class template provides the core event demultiplexing
/// functionality for users of the asynchronous I/O objects, such as
/// stream_socket, and also to developers of custom asynchronous services. Most
/// applications will use the demuxer typedef.
template <typename Demuxer_Service>
class basic_demuxer
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide demuxer operations.
  typedef Demuxer_Service service_type;

  /// Default constructor.
  basic_demuxer()
    : service_registry_(*this),
      service_(get_service(service_factory<Demuxer_Service>()))
  {
#if defined(_WIN32)
    detail::winsock_init<>::use();
#else
    detail::signal_init<>::use();
#endif // _WIN32
  }

  /// Run the demuxer's event processing loop.
  /**
   * The run function blocks until all operations have completed and there are
   * no more completions to be delivered, or until the demuxer has been
   * interrupted. The run function may be safely called again once it has
   * completed to execute any new operations or deliver new completions.
   *
   * Multiple threads may call the run function to set up a pool of threads
   * from which the demuxer may dispatch the completion notifications.
   */
  void run()
  {
    service_.run();
  }

  /// Interrupt the demuxer's event processing loop.
  /**
   * This function does not block, but instead simply signals to the demuxer
   * that all invocations of its run member function should return as soon as
   * possible.
   *
   * Note that if the run function is interrupted and is not called again later
   * then its operations may not have finished and completions not delivered.
   * In this case a demuxer implementation is not required to make any
   * guarantee that any resources associated with those operations would be
   * cleaned up.
   */
  void interrupt()
  {
    service_.interrupt();
  }

  /// Reset the demuxer in preparation for a subsequent run invocation.
  /**
   * This function must be called prior to any second or later set of
   * invocations of the run function. It allows the demuxer to reset any
   * internal state, such as an interrupt flag.
   *
   * This function must not be called while there are any unfinished calls to
   * the run function.
   */
  void reset()
  {
    service_.reset();
  }

  /// Notify the demuxer that an operation has started.
  /**
   * This function is used to inform the demuxer that a new operation has
   * begun. A call to this function must be matched with a later corresponding
   * call to operation_completed.
   */
  void operation_started()
  {
    service_.operation_started();
  }

  /// Notify the demuxer that an operation has completed.
  /**
   * This function is used to inform the demuxer that an operation has
   * completed and that it should invoke the given completion handler. A call
   * to this function must be matched with an earlier corresponding call to
   * operation_started.
   *
   * The completion handler is guaranteed to be called only from a thread in
   * which the run member function is being invoked. 
   *
   * @param handler The completion handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void operation_completed(Handler handler)
  {
    service_.operation_completed(handler, null_completion_context(), false);
  }

  /// Notify the demuxer that an operation has completed.
  /**
   * This function is used to inform the demuxer that an operation has
   * completed and that it should invoke the given completion handler. A call
   * to this function must be matched with an earlier corresponding call to
   * operation_started.
   *
   * The completion handler is guaranteed to be called only from a thread in
   * which the run member function is being invoked. 
   *
   * @param handler The completion handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Copies will be made
   * of the context object as required, however all copies are equivalent.
   */
  template <typename Handler, typename Completion_Context>
  void operation_completed(Handler handler, Completion_Context context)
  {
    service_.operation_completed(handler, context, false);
  }

  /// Notify the demuxer that an operation has completed.
  /**
   * This function is used to inform the demuxer that an operation has
   * completed and that it should invoke the given completion handler. A call
   * to this function must be matched with an earlier corresponding call to
   * operation_started.
   *
   * The completion handler is guaranteed to be called only from a thread in
   * which the run member function is being invoked. 
   *
   * @param handler The completion handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Copies will be made
   * of the context object as required, however all copies are equivalent.
   *
   * @param allow_nested_delivery If true, this allows the demuxer to run the
   * completion handler before operation_completed returns, as an optimisation.
   * This is at the discretion of the demuxer implementation, but may only be
   * done if it can meet the guarantee that the handler is invoked from a
   * thread executing the run function. If false, the function returns
   * immediately.
   */
  template <typename Handler, typename Completion_Context>
  void operation_completed(Handler handler, Completion_Context context,
      bool allow_nested_delivery)
  {
    service_.operation_completed(handler, context, allow_nested_delivery);
  }

  /// Notify the demuxer of an operation that started and finished immediately.
  /**
   * This function is used to inform the demuxer that an operation has both
   * started and completed immediately, and that it should invoke the given
   * completion handler. A call to this function must not have either of a
   * corresponding operation_started or operation_completed.
   *
   * The completion handler is guaranteed to be called only from a thread in
   * which the run member function is being invoked. 
   *
   * @param handler The completion handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void operation_immediate(Handler handler)
  {
    service_.operation_immediate(handler, null_completion_context(), false);
  }

  /// Notify the demuxer of an operation that started and finished immediately.
  /**
   * This function is used to inform the demuxer that an operation has both
   * started and completed immediately, and that it should invoke the given
   * completion handler. A call to this function must not have either of a
   * corresponding operation_started or operation_completed.
   *
   * The completion handler is guaranteed to be called only from a thread in
   * which the run member function is being invoked. 
   *
   * @param handler The completion handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Copies will be made
   * of the context object as required, however all copies are equivalent.
   */
  template <typename Handler, typename Completion_Context>
  void operation_immediate(Handler handler, Completion_Context context)
  {
    service_.operation_immediate(handler, context, false);
  }

  /// Notify the demuxer of an operation that started and finished immediately.
  /**
   * This function is used to inform the demuxer that an operation has both
   * started and completed immediately, and that it should invoke the given
   * completion handler. A call to this function must not have either of a
   * corresponding operation_started or operation_completed.
   *
   * The completion handler is guaranteed to be called only from a thread in
   * which the run member function is being invoked. 
   *
   * @param handler The completion handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Copies will be made
   * of the context object as required, however all copies are equivalent.
   *
   * @param allow_nested_delivery If true, this allows the demuxer to run the
   * completion handler before operation_immediate returns, as an optimisation.
   * This is at the discretion of the demuxer implementation, but may only be
   * done if it can meet the guarantee that the handler is invoked from a
   * thread executing the run function. If false, the function returns
   * immediately.
   */
  template <typename Handler, typename Completion_Context>
  void operation_immediate(Handler handler, Completion_Context context,
      bool allow_nested_delivery)
  {
    service_.operation_immediate(handler, context, allow_nested_delivery);
  }

  /// Obtain the service interface corresponding to the given type.
  /**
   * This function is used to locate a service interface that corresponds to
   * the given service type. If there is no existing implementation of the
   * service, then the demuxer will use the supplied factory to create a new
   * instance.
   *
   * @param factory The factory to use to create the service.
   *
   * @return The service interface implementing the specified service type.
   * Ownership of the service interface is not transferred to the caller.
   */
  template <typename Service>
  Service& get_service(service_factory<Service> factory)
  {
    return service_registry_.get_service(factory);
  }

private:
  /// The service registry.
  detail::service_registry<basic_demuxer<Demuxer_Service> > service_registry_;

  /// The underlying demuxer service implementation.
  Demuxer_Service& service_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DEMUXER_HPP

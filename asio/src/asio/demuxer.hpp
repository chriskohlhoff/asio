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

#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include "asio/completion_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

class demuxer_task;
class demuxer_service;
class service;
class service_provider_factory;
class service_type_id;
namespace detail { class service_registry; }

/// The demuxer class provides the core event demultiplexing functionality for
/// both users of the asynchronous I/O objects, such as stream_socket, and to
/// developers of custom service providers.
class demuxer
  : private boost::noncopyable
{
public:
  /// Default constructor.
  /**
   * This constructor may be used to create a demuxer object that uses the
   * default service provider factory.
   */
  demuxer();

  /// Construct a demuxer object using the specified service provider factory.
  /**
   * This constructor may be used to supply a specific service provider factory
   * to the demuxer, so that custom service providers can be created on demand.
   * If a demuxer is asked, via the get_service function, to supply a service
   * interface, and no provider is currently registered for that service, then
   * the demuxer will use the factory to create a new service provider
   * implementation.
   *
   * @param factory The supplied factory that may be used to create a service
   * provider that implements a particular interface. The caller retains
   * ownership of the factory object, and is responsible for ensuring that the
   * factory object has a longer lifetime than the demuxer object.
   */
  explicit demuxer(service_provider_factory& factory);

  /// Destructor.
  ~demuxer();

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
  void run();

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
  void interrupt();

  /// Reset the demuxer in preparation for a subsequent run invocation.
  /**
   * This function must be called prior to any second or later set of
   * invocations of the run function. It allows the demuxer to reset any
   * internal state, such as an interrupt flag.
   *
   * This function must not be called while there are any unfinished calls to
   * the run function.
   */
  void reset();

  /// Add a task to the demuxer.
  /**
   * This function may be used to instruct the demuxer to execute a task until
   * it has finished. A demuxer implementation may choose to execute a task in
   * any thread, including a thread from which the run function is being
   * invoked.
   *
   * @param task The task to be added to the demuxer. The caller retains
   * ownership of the task object, and is responsible for ensuring that the
   * task object is not destroyed until the task has finished execution and has
   * been instructed to clean up.
   *
   * @param arg A caller-defined token to be passed to all demuxer_task virtual
   * function invocations.
   */
  void add_task(demuxer_task& task, void* arg);

  /// Notify the demuxer that an operation has started.
  /**
   * This function is used to inform the demuxer that a new operation has
   * begun. A call to this function must be matched with a later corresponding
   * call to operation_completed.
   */
  void operation_started();

  /// The type of a handler to be called when a completion is delivered.
  typedef boost::function0<void> completion_handler;

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
   * a copy of the handler object as required.
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Ownership of the
   * completion_context object is retained by the caller, which must guarantee
   * that it is valid until after the handler has been called.
   *
   * @param allow_nested_delivery If true, this allows the demuxer to run the
   * completion handler before operation_completed returns, as an optimisation.
   * This is at the discretion of the demuxer implementation, but may only be
   * done if it can meet the guarantee that the handler is invoked from a
   * thread executing the run function. If false, the function returns
   * immediately.
   */
  void operation_completed(const completion_handler& handler,
      completion_context& context = completion_context::null(),
      bool allow_nested_delivery = false);

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
   * a copy of the handler object as required.
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Ownership of the
   * completion_context object is retained by the caller, which must guarantee
   * that it is valid until after the handler has been called.
   *
   * @param allow_nested_delivery If true, this allows the demuxer to run the
   * completion handler before operation_immediate returns, as an optimisation.
   * This is at the discretion of the demuxer implementation, but may only be
   * done if it can meet the guarantee that the handler is invoked from a
   * thread executing the run function. If false, the function returns
   * immediately.
   */
  void operation_immediate(const completion_handler& handler,
      completion_context& context = completion_context::null(),
      bool allow_nested_delivery = false);

  /// Obtain the service interface corresponding to the given type.
  /**
   * This function is used to locate a service interface that corresponds to
   * the given service type. If no existing provider implements the service
   * type then the demuxer will use the service provider factory that was
   * specified when it was constructed to create a new provider.
   *
   * @param service_type The unique identifier of a service.
   *
   * @return The service interface implementing the specified service type.
   * Ownership of the service interface is not transferred to the caller.
   *
   * @throws service_unavailable if no provider is able to provide the
   * requested service.
   */
  service& get_service(const service_type_id& service_type);

private:
  /// The service registry.
  detail::service_registry* service_registry_;

  /// The underlying demuxer service implementation.
  demuxer_service& service_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEMUXER_HPP

//
// Dispatcher.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Dispatcher concept.
/**
 * @par Implemented By:
 * asio::basic_demuxer @n
 * asio::basic_locking_dispatcher
 */
class Dispatcher
{
public:
  /// Request the dispatcher to invoke the given handler.
  /**
   * This function is used to ask the dispatcher to execute the given handler.
   *
   * @param handler The handler to be called. The dispatcher will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void dispatch(Handler handler);

  /// Request the dispatcher to invoke the given handler and return
  /// immediately.
  /**
   * This function is used to ask the dispatcher to execute the given handler,
   * but without allowing the dispatcher to call the handler from inside this
   * function.
   *
   * @param handler The handler to be called. The dispatcher will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void post(Handler handler);

  /// Create a new handler that automatically dispatches the wrapped handler
  /// on the dispatcher.
  /**
   * This function is used to create a new handler function object that, when
   * invoked, will automatically pass the wrapped handler to the dispatcher's
   * dispatch() function.
   *
   * @param handler The handler to be wrapped. The dispatcher will make a copy
   * of the handler object as required. The equivalent function signature of
   * the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  implementation_defined wrap(Handler handler);
};

//
// local/basic_pipe.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern ( klemens dot morgenstern at gmx dot net )
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_PIPE_HPP
#define ASIO_BASIC_PIPE_HPP

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/windows/basic_stream_handle.hpp"
# include "asio/windows/connect_pipe.hpp"
# include "asio/windows/devices.hpp"
#else
# include "asio/posix/basic_stream_descriptor.hpp"
# include "asio/posix/connect_pipe.hpp"
# include "asio/posix/devices.hpp"
#endif


namespace asio
{


#if !defined(GENERATING_DOCUMENTATION)
#if defined(ASIO_HAS_IOCP)

namespace detail {using detail::windows::connect_pipe;}

using detail::windows::open_null_reader;
using detail::windows::open_null_writer;
using detail::windows::open_stdin;
using detail::windows::open_stdout;
using detail::windows::open_stderr;

#else

namespace detail {using detail::posix::connect_pipe;}

using detail::posix::open_null_reader;
using detail::posix::open_null_writer;
using detail::posix::open_stdin;
using detail::posix::open_stdout;
using detail::posix::open_stderr;

#endif




template<typename Executor = any_io_executor>
struct basic_pipe_read_end :
#if defined(ASIO_HAS_IOCP)
         windows::basic_stream_handle<Executor>
#else
         posix::basic_stream_descriptor<Executor>
#endif
{
#if defined(ASIO_HAS_IOCP)
    typedef windows::basic_stream_handle<Executor> stream_type;
#else
    typedef posix::basic_stream_descriptor<Executor> stream_type;
#endif

    template <typename Executor1>
    struct rebind_executor
    {
        typedef basic_pipe_read_end<Executor1> other;
    };

    using stream_type::read_some;
    using stream_type::async_read_some;
    using stream_type::assign;
    using stream_type::native_handle;
    using stream_type::get_executor;
    using stream_type::is_open;
    using stream_type::close;
    using stream_type::cancel;
    using stream_type::lowest_layer;

    using stream_type::stream_type;
    using stream_type::operator=;

    typedef typename stream_type::executor_type executor_type;
    typedef typename stream_type::lowest_layer_type lowest_layer_type;

};


template<typename Executor = any_io_executor>
struct basic_pipe_write_end :
#if defined(ASIO_HAS_IOCP)
    windows::basic_stream_handle<Executor>
#else
    posix::basic_stream_descriptor<Executor>
#endif
{
#if defined(ASIO_HAS_IOCP)
    typedef windows::basic_stream_handle<Executor> stream_type;
#else
    typedef posix::basic_stream_descriptor<Executor> stream_type;
#endif

    template <typename Executor1>
    struct rebind_executor
    {
        typedef basic_pipe_write_end<Executor1> other;
    };
    using stream_type::write_some;
    using stream_type::async_write_some;

    using stream_type::assign;
    using stream_type::native_handle;
    using stream_type::get_executor;
    using stream_type::is_open;
    using stream_type::close;
    using stream_type::cancel;
    using stream_type::lowest_layer;

    using stream_type::stream_type;
    using stream_type::operator=;

    typedef typename stream_type::executor_type executor_type;
    typedef typename stream_type::lowest_layer_type lowest_layer_type;
};

#endif


/// Provides pipe async communication functionality.
/**
 * The pipe class template provides asynchronous and
 * blocking stream-oriented pipe functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * Synchronous @c read_some and @c write_some operations are thread safe with
 * respect to each other, if the underlying operating system calls are also
 * thread safe. This means that it is permitted to perform concurrent calls to
 * these synchronous operations on a single pipe object. Other synchronous
 * operations, such as @c close, are not thread safe.
 *
 * @par Concepts:
 * AsyncReadStream, AsyncWriteStream, Stream, SyncReadStream, SyncWriteStream.
 */
template<typename Executor = any_io_executor>
struct basic_pipe
{
    /// The representation of a read end.
    typedef basic_pipe_read_end<Executor>   read_end_type;
    /// The representation of a write end.
    typedef basic_pipe_write_end<Executor> write_end_type;

    /// The read end.
    read_end_type   read_end;
    /// The write end.
    write_end_type write_end;

    /// The native representation of a descriptor.
    typedef typename read_end_type::native_handle_type native_handle_type;

    /// The type of the executor associated with the object.
    typedef Executor executor_type;


    /// Rebinds the pipe type to another executor.
    template <typename Executor1>
    struct rebind_executor
    {
        /// The pipe type when rebound to the specified executor.
        typedef basic_pipe<Executor1> other;
    };

    /// Construct a pipe and connect it.
    /**
     * This constructor creates a pipe and opens it.
     *
     * @param ex The I/O executor that the pipe will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe.
     */
    basic_pipe(Executor exec) : read_end(exec), write_end(exec)
    {
        detail::connect_pipe(read_end, write_end);
    }

    /// Construct a named pipe and connect it.
    /**
     * This constructor creates a named pipe and connects it.
     *
     * @param ex The I/O executor that the pipe will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe.
     *
     * @param filename The path of the pipe. Note that operating systems constraint the names one can choopse.
     */
#if defined(__cpp_lib_filesystem) || defined(GENERATING_DOCUMENATION)
    basic_pipe(Executor exec, const std::filesystem::path & filename) : read_end(exec), write_end(exec)
    {
        detail::connect_pipe(filename, read_end, write_end);
    }
#endif
    /// Construct a pipe and connect it.
    /**
     * This constructor creates a pipe and opens it.
     *
     * @param ex The I/O executor that the pipe will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe.
     */
    template <typename ExecutionContext>
    explicit basic_pipe(ExecutionContext& context,
                        typename constraint<
                                is_convertible<ExecutionContext&, execution_context&>::value,
                                defaulted_constraint
                        >::type = defaulted_constraint()) : basic_pipe(context.get_executor()) { }

    /// Construct a named pipe and connect it.
    /**
     * This constructor creates a named pipe and connects it.
     *
     * @param ex The I/O executor that the pipe will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe.
     *
     * @param filename The path of the pipe. Note that operating systems constraint the names one can choopse.
     */
#if defined(__cpp_lib_filesystem) || defined(GENERATING_DOCUMENATION)
    template <typename ExecutionContext>
    explicit basic_pipe(ExecutionContext& context,
                        const std::filesystem::path & filename,
                        typename constraint<
                                is_convertible<ExecutionContext&, execution_context&>::value,
                                defaulted_constraint
                        >::type = defaulted_constraint()) : basic_pipe(context.get_executor(), filename) { }
#endif
    /// Move-construct a pipe from another.
    /**
     * This constructor moves a pipe from one object to another.
     *
     * @param other The other pipe object from which the move will occur.
     *
     * @note Following the move, the moved-from object is closed.
     */
    basic_pipe(basic_pipe && ) = default;
    /// Move-assign a pipe from another.
    /**
     * This assignment operator moves a pipe from one object to
     * another.
     *
     * @param other The other pipe object from which the move will occur.

     *
     * @note Following the move, the moved-from object is closed.
     */
    basic_pipe & operator=(basic_pipe && ) = default;

    /// Read some data from the pipe.
    /**
     * This function is used to read data from the pipe. The function
     * call will block until one or more bytes of data has been read successfully,
     * or until an error occurs.
     *
     * @param buffers One or more buffers into which the data will be read.
     *
     * @returns The number of bytes read.
     *
     * @throws asio::system_error Thrown on failure. An error code of
     * asio::error::eof indicates that the connection was closed by the
     * peer.
     *
     * @note The read_some operation may not read all of the requested number of
     * bytes. Consider using the @ref read function if you need to ensure that
     * the requested amount of data is read before the blocking operation
     * completes.
     *
     * @par Example
     * To read into a single data buffer use the @ref buffer function as follows:
     * @code
     * pipe.read_some(asio::buffer(data, size));
     * @endcode
     * See the @ref buffer documentation for information on reading into multiple
     * buffers in one go, and how to use it with arrays, boost::array or
     * std::vector.
     */
    template <typename MutableBufferSequence>
    std::size_t read_some(const MutableBufferSequence& buffers) {return read_end.read_some(buffers);}

    /// Read some data from the pipe.
    /**
     * This function is used to read data from the pipe. The function
     * call will block until one or more bytes of data has been read successfully,
     * or until an error occurs.
     *
     * @param buffers One or more buffers into which the data will be read.
     *
     * @param ec Set to indicate what error occurred, if any.
     *
     * @returns The number of bytes read. Returns 0 if an error occurred.
     *
     * @note The read_some operation may not read all of the requested number of
     * bytes. Consider using the @ref read function if you need to ensure that
     * the requested amount of data is read before the blocking operation
     * completes.
     */
    template <typename MutableBufferSequence>
    std::size_t read_some(const MutableBufferSequence& buffers, asio::error_code& ec) {return read_end.read_some(buffers, ec);}

    /// Start an asynchronous read.
    /**
     * This function is used to asynchronously read data from the stream
     * descriptor. The function call always returns immediately.
     *
     * @param buffers One or more buffers into which the data will be read.
     * Although the buffers object may be copied as necessary, ownership of the
     * underlying memory blocks is retained by the caller, which must guarantee
     * that they remain valid until the handler is called.
     *
     * @param handler The handler to be called when the read operation completes.
     * Copies will be made of the handler as required. The function signature of
     * the handler must be:
     * @code void handler(
     *   const asio::error_code& error, // Result of operation.
     *   std::size_t bytes_transferred           // Number of bytes read.
     * ); @endcode
     * Regardless of whether the asynchronous operation completes immediately or
     * not, the handler will not be invoked from within this function. On
     * immediate completion, invocation of the handler will be performed in a
     * manner equivalent to using asio::post().
     *
     * @note The read operation may not read all of the requested number of bytes.
     * Consider using the @ref async_read function if you need to ensure that the
     * requested amount of data is read before the asynchronous operation
     * completes.
     *
     * @par Example
     * To read into a single data buffer use the @ref buffer function as follows:
     * @code
     * pipe.async_read_some(asio::buffer(data, size), handler);
     * @endcode
     * See the @ref buffer documentation for information on reading into multiple
     * buffers in one go, and how to use it with arrays, boost::array or
     * std::vector.
     */
    template <typename MutableBufferSequence,
            ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
                                              std::size_t)) ReadHandler
            ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
    ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
                                 void (asio::error_code, std::size_t))
    async_read_some(const MutableBufferSequence& buffers,
                    ASIO_MOVE_ARG(ReadHandler) handler
                    ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
    {
        return read_end.async_read_some(buffers, ASIO_MOVE_CAST(ReadHandler)(handler));
    }

    /// Write some data to the pipe.
    /**
     * This function is used to write data to the pipe. The function
     * call will block until one or more bytes of the data has been written
     * successfully, or until an error occurs.
     *
     * @param buffers One or more data buffers to be written to the pipe.
     *
     * @returns The number of bytes written.
     *
     * @throws asio::system_error Thrown on failure. An error code of
     * asio::error::eof indicates that the connection was closed by the
     * peer.
     *
     * @note The write_some operation may not transmit all of the data to the
     * peer. Consider using the @ref write function if you need to ensure that
     * all data is written before the blocking operation completes.
     *
     * @par Example
     * To write a single data buffer use the @ref buffer function as follows:
     * @code
     * pipe.write_some(asio::buffer(data, size));
     * @endcode
     * See the @ref buffer documentation for information on writing multiple
     * buffers in one go, and how to use it with arrays, boost::array or
     * std::vector.
     */
    template <typename ConstBufferSequence>
    std::size_t write_some(const ConstBufferSequence& buffers)
    {
        return write_end.write_some(buffers);
    }

    /// Write some data to the pipe.
    /**
     * This function is used to write data to the pipe. The function
     * call will block until one or more bytes of the data has been written
     * successfully, or until an error occurs.
     *
     * @param buffers One or more data buffers to be written to the pipe.
     *
     * @param ec Set to indicate what error occurred, if any.
     *
     * @returns The number of bytes written. Returns 0 if an error occurred.
     *
     * @note The write_some operation may not transmit all of the data to the
     * peer. Consider using the @ref write function if you need to ensure that
     * all data is written before the blocking operation completes.
     */
    template <typename ConstBufferSequence>
    std::size_t write_some(const ConstBufferSequence& buffers, asio::error_code& ec)
    {
        return write_end.write_some(buffers, ec);
    }

    /// Start an asynchronous write.
    /**
     * This function is used to asynchronously write data to the stream
     * pipe. The function call always returns immediately.
     *
     * @param buffers One or more data buffers to be written to the pipe.
     * Although the buffers object may be copied as necessary, ownership of the
     * underlying memory blocks is retained by the caller, which must guarantee
     * that they remain valid until the handler is called.
     *
     * @param handler The handler to be called when the write operation completes.
     * Copies will be made of the handler as required. The function signature of
     * the handler must be:
     * @code void handler(
     *   const asio::error_code& error, // Result of operation.
     *   std::size_t bytes_transferred           // Number of bytes written.
     * ); @endcode
     * Regardless of whether the asynchronous operation completes immediately or
     * not, the handler will not be invoked from within this function. On
     * immediate completion, invocation of the handler will be performed in a
     * manner equivalent to using asio::post().
     *
     * @note The write operation may not transmit all of the data to the peer.
     * Consider using the @ref async_write function if you need to ensure that all
     * data is written before the asynchronous operation completes.
     *
     * @par Example
     * To write a single data buffer use the @ref buffer function as follows:
     * @code
     * descriptor.async_write_some(asio::buffer(data, size), handler);
     * @endcode
     * See the @ref buffer documentation for information on writing multiple
     * buffers in one go, and how to use it with arrays, boost::array or
     * std::vector.
     */
    template <typename ConstBufferSequence,
            ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code, std::size_t)) WriteHandler
            ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
    ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler, void (asio::error_code, std::size_t))
    async_write_some(const ConstBufferSequence& buffers,
                     ASIO_MOVE_ARG(WriteHandler) handler  ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
    {
        return write_end.async_write_some(buffers, ASIO_MOVE_CAST(WriteHandler)(handler));
    }

    /// Close the pipe.
    /**
     * This function is used to close the pipe. Any asynchronous read or
     * write operations will be cancelled immediately, and will complete with the
     * asio::error::operation_aborted error.
     *
     * @throws asio::system_error Thrown on failure. Note that, even if
     * the function indicates an error, the underlying descriptor is closed.
     */
    void close()
    {
        read_end.close();
        write_end.close();
    }
    /// Close the pipe.
    /**
     * This function is used to close the pipe. Any asynchronous read or
     * write operations will be cancelled immediately, and will complete with the
     * asio::error::operation_aborted error.
     *
     * @param ec Set to indicate what error occurred, if any. Note that, even if
     * the function indicates an error, the underlying descriptor is closed.
     */
    void close(error_code & ec) ASIO_NOEXCEPT
    {
        read_end.close(ec);
        write_end.close(ec);
    }
    /// Cancel all asynchronous operations associated with the pipe.
    /**
     * This function causes all outstanding asynchronous read or write operations
     * to finish immediately, and the handlers for cancelled operations will be
     * passed the asio::error::operation_aborted error.
     *
     * @throws asio::system_error Thrown on failure.
     */
    void cancel()
    {
        read_end.cancel();
        write_end.cancel();
    }

    /// Cancel all asynchronous operations associated with the pipe.
    /**
     * This function causes all outstanding asynchronous read or write operations
     * to finish immediately, and the handlers for cancelled operations will be
     * passed the asio::error::operation_aborted error.
     *
     * @param ec Set to indicate what error occurred, if any.
     */
    void cancel(asio::error_code& ec)
    {
        read_end.cancel(ec);
        write_end.cancel(ec);
    }

    /// Determine whether the pipe is open.
    bool is_open() ASIO_NOEXCEPT
    {
        return read_end.is_open() && write_end.is_open();
    }
};


#if defined(GENERATING_DOCUMENTATION)

/// Open a reader for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 * @param pipe_end The pipe end to be connected to the null-device
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @note Every call to this functions opens the pseudo-file again.
 */
template<typename Executor>
void open_null_reader(basic_pipe_read_end <Executor> &pipe_end, error_code &ec) noexcept;

/// Open a reader for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The pipe end hat was connected to the null-device
 *
 * @note Every call to this functions opens the pseudo-file again.
 */
template<typename Executor>
void open_null_reader(basic_pipe_read_end <Executor> &pipe_end);

/// Open a reader for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 * @param executor The executor of the pipe
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 */
template<typename Executor>
inline basic_pipe_read_end <Executor> open_null_reader(Executor executor, error_code &ec) noexcept;

/// Open a reader for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 * @param executor The executor of the pipe
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 */
template<typename Executor>
basic_pipe_read_end <Executor> open_null_reader(Executor executor);


/// Open a writer for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 * @param pipe_end The pipe end to be connected to the null-device
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @note Every call to this functions opens the pseudo-file again.
 */
template<typename Executor>
void open_null_writer(basic_pipe_write_end <Executor> &pipe_end, error_code &ec) noexcept;

/// Open a writer for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The pipe end hat was connected to the null-device
 *
 * @note Every call to this functions opens the pseudo-file again.
 */
template<typename Executor>
void open_null_writer(basic_pipe_write_end <Executor> &pipe_end);

/// Open a writer for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 * @param executor The executor of the pipe
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 */
template<typename Executor>
inline basic_pipe_write_end<Executor> open_null_writer(Executor executor, error_code &ec) noexcept;

/// Open a writer for the null device. This is `/dev/null` on posix and `NUL` on windows.
/**
 * @param executor The executor of the pipe
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 */
template<typename Executor>
basic_pipe_write_end<Executor> open_null_writer(Executor executor);



/// Open a read end for stdin.
/**
 * @param pipe_end The pipe end to be connected to the null-device
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @note Every call to this functions opens the pseudo-file again.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
void open_stdin(basic_pipe_read_end <Executor> &pipe_end, error_code &ec) noexcept;

/// Open a read end for stdin.
/**
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The pipe end hat was connected to the null-device
 *
 * @note Every call to this functions opens the pseudo-file again.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
void open_stdin(basic_pipe_read_end <Executor> &pipe_end);

/// Open a read end for stdin.
/**
 * @param executor The executor of the pipe
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
inline basic_pipe_read_end <Executor> open_stdin(Executor executor, error_code &ec) noexcept;

/// Open a read end for stdin.
/**
 * @param executor The executor of the pipe
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
basic_pipe_read_end <Executor> open_stdin(Executor executor);




/// Open a write end for stdout.
/**
 * @param pipe_end The pipe end to be connected to the null-device
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @note Every call to this functions opens the pseudo-file again.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
void open_stdout(basic_pipe_write_end <Executor> &pipe_end, error_code &ec) noexcept;

/// Open a write end for stdout.
/**
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The pipe end hat was connected to the null-device
 *
 * @note Every call to this functions opens the pseudo-file again.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
void open_stdout(basic_pipe_write_end <Executor> &pipe_end);

/// Open a write end for stdout.
/**
 * @param executor The executor of the pipe
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
inline basic_pipe_write_end <Executor> open_stdout(Executor executor, error_code &ec) noexcept;

/// Open a write end for stdout.
/**
 * @param executor The executor of the pipe
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
basic_pipe_write_end <Executor> open_stdout(Executor executor);


/// Open a write end for stderr.
/**
 * @param pipe_end The pipe end to be connected to the null-device
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @note Every call to this functions opens the pseudo-file again.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
void open_stderr(basic_pipe_write_end <Executor> &pipe_end, error_code &ec) noexcept;

/// Open a write end for stderr.
/**
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The pipe end hat was connected to the null-device
 *
 * @note Every call to this functions opens the pseudo-file again.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
void open_stderr(basic_pipe_write_end <Executor> &pipe_end);

/// Open a write end for stderr.
/**
 * @param executor The executor of the pipe
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
inline basic_pipe_write_end <Executor> open_stderr(Executor executor, error_code &ec) noexcept;

/// Open a write end for stderr.
/**
 * @param executor The executor of the pipe
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @returns The connected pipe.
 *
 * @note Every call to this functions opens the pseudo-file again.
 *
 * @note Only available in C++11.
 * 
 * @warning This function might only work once on Windows.
 */
template<typename Executor>
basic_pipe_write_end <Executor> open_stderr(Executor executor);


/// Provides pipe_read_end async communication functionality.
/**
 * The pipe_read_end class template provides asynchronous and
 * blocking stream-oriented pipe_read_end functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * Synchronous @c read_some and @c write_some operations are thread safe with
 * respect to each other, if the underlying operating system calls are also
 * thread safe. This means that it is permitted to perform concurrent calls to
 * these synchronous operations on a single pipe_read_end object. Other synchronous
 * operations, such as @c close, are not thread safe.
 *
 * @par Concepts:
 * AsyncReadStream, Stream, SyncWriteStream.
 */
template<typename Executor = any_io_executor>
struct basic_pipe_read_end
{
    /// The used stream type.
    typedef implementation_defined stream_type;

    /// The native representation of a descriptor.
    typedef implementation_defined native_handle_type;

    /// The type of the executor associated with the object.
    typedef Executor executor_type;


    /// Rebinds the pipe_read_end type to another executor.
    template <typename Executor1>
    struct rebind_executor
    {
        /// The pipe_read_end type when rebound to the specified executor.
        typedef basic_pipe_read_end<Executor1> other;
    };

    /// Construct a pipe_read_end and doesn't open it.
    /**
     * This constructor creates a pipe_read_end doesn't open it.
     *
     * @param ex The I/O executor that the pipe_read_end will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe_read_end.
     */
    basic_pipe_read_end(Executor exec);

    /// Construct a pipe_read_end and doesn't open it.
    /**
     * This constructor creates a pipe_read_end and doesn't open it.
     *
     * @param ex The I/O executor that the pipe_read_end will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe_read_end.
     */
    template <typename ExecutionContext>
    explicit basic_pipe_read_end(ExecutionContext& context);

    /// Move-construct a pipe_read_end from another.
    /**
     * This constructor moves a pipe_read_end from one object to another.
     *
     * @param other The other pipe_read_end object from which the move will occur.
     *
     * @note Following the move, the moved-from object is closed.
     */
    basic_pipe_read_end(basic_pipe_read_end && ) = default;
    /// Move-assign a pipe_read_end from another.
    /**
     * This assignment operator moves a pipe_read_end from one object to
     * another.
     *
     * @param other The other pipe_read_end object from which the move will occur.

     *
     * @note Following the move, the moved-from object is closed.
     */
    basic_pipe_read_end & operator=(basic_pipe_read_end && ) = default;

    /// Read some data from the pipe_read_end.
    /**
     * This function is used to read data from the pipe_read_end. The function
     * call will block until one or more bytes of data has been read successfully,
     * or until an error occurs.
     *
     * @param buffers One or more buffers into which the data will be read.
     *
     * @returns The number of bytes read.
     *
     * @throws asio::system_error Thrown on failure. An error code of
     * asio::error::eof indicates that the connection was closed by the
     * peer.
     *
     * @note The read_some operation may not read all of the requested number of
     * bytes. Consider using the @ref read function if you need to ensure that
     * the requested amount of data is read before the blocking operation
     * completes.
     *
     * @par Example
     * To read into a single data buffer use the @ref buffer function as follows:
     * @code
     * pipe_read_end.read_some(asio::buffer(data, size));
     * @endcode
     * See the @ref buffer documentation for information on reading into multiple
     * buffers in one go, and how to use it with arrays, boost::array or
     * std::vector.
     */
    template <typename MutableBufferSequence>
    std::size_t read_some(const MutableBufferSequence& buffers);

    /// Read some data from the pipe_read_end.
    /**
     * This function is used to read data from the pipe_read_end. The function
     * call will block until one or more bytes of data has been read successfully,
     * or until an error occurs.
     *
     * @param buffers One or more buffers into which the data will be read.
     *
     * @param ec Set to indicate what error occurred, if any.
     *
     * @returns The number of bytes read. Returns 0 if an error occurred.
     *
     * @note The read_some operation may not read all of the requested number of
     * bytes. Consider using the @ref read function if you need to ensure that
     * the requested amount of data is read before the blocking operation
     * completes.
     */
    template <typename MutableBufferSequence>
    std::size_t read_some(const MutableBufferSequence& buffers, asio::error_code& ec) {return read_end.read_some(buffers, ec);}

      /// Start an asynchronous read.
  /**
   * This function is used to asynchronously read data from the stream
   * descriptor. The function call always returns immediately.
   *
   * @param buffers One or more buffers into which the data will be read.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred           // Number of bytes read.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. On
   * immediate completion, invocation of the handler will be performed in a
   * manner equivalent to using asio::post().
   *
   * @note The read operation may not read all of the requested number of bytes.
   * Consider using the @ref async_read function if you need to ensure that the
   * requested amount of data is read before the asynchronous operation
   * completes.
   *
   * @par Example
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code
   * pipe_end.async_read_some(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
    template <typename MutableBufferSequence,
            ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
                                              std::size_t)) ReadHandler
            ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
    ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
                                 void (asio::error_code, std::size_t))
    async_read_some(const MutableBufferSequence& buffers,
                    ASIO_MOVE_ARG(ReadHandler) handler
                    ASIO_DEFAULT_COMPLETION_TOKEN(executor_type));

    /// Close the pipe_read_end.
    /**
     * This function is used to close the pipe_read_end. Any asynchronous read or
     * write operations will be cancelled immediately, and will complete with the
     * asio::error::operation_aborted error.
     *
     * @throws asio::system_error Thrown on failure. Note that, even if
     * the function indicates an error, the underlying descriptor is closed.
     */
    void close();

    /// Close the pipe_read_end.
    /**
     * This function is used to close the pipe_read_end. Any asynchronous read or
     * write operations will be cancelled immediately, and will complete with the
     * asio::error::operation_aborted error.
     *
     * @param ec Set to indicate what error occurred, if any. Note that, even if
     * the function indicates an error, the underlying descriptor is closed.
     */
    void close(error_code & ec) noexcept;

    /// Cancel all asynchronous operations associated with the pipe_read_end.
    /**
     * This function causes all outstanding asynchronous read or write operations
     * to finish immediately, and the handlers for cancelled operations will be
     * passed the asio::error::operation_aborted error.
     *
     * @throws asio::system_error Thrown on failure.
     */
    void cancel();

    /// Cancel all asynchronous operations associated with the pipe_read_end.
    /**
     * This function causes all outstanding asynchronous read or write operations
     * to finish immediately, and the handlers for cancelled operations will be
     * passed the asio::error::operation_aborted error.
     *
     * @param ec Set to indicate what error occurred, if any.
     */
    void cancel(asio::error_code& ec);

    /// Determine whether the pipe_read_end is open.
    bool is_open() noexcept;

    /// Get the native pipe end representation.
    /**
     * This function may be used to obtain the underlying representation of the
     * descriptor. This is intended to allow access to native descriptor
     * functionality that is not otherwise provided.
     */
    native_handle_type native_handle() const;

    /// Get a reference to the lowest layer.
    /**
     * This function returns a reference to the lowest layer in a stack of
     * layers. Since a descriptor cannot contain any further layers, it
     * simply returns a reference to itself.
     *
     * @return A reference to the lowest layer in the stack of layers. Ownership
     * is not transferred to the caller.
     */
    lowest_layer_type& lowest_layer();

    /// Get a const reference to the lowest layer.
    /**
     * This function returns a const reference to the lowest layer in a stack of
     * layers. Since a descriptor cannot contain any further layers, it
     * simply returns a reference to itself.
     *
     * @return A const reference to the lowest layer in the stack of layers.
     * Ownership is not transferred to the caller.
     */
    const lowest_layer_type& lowest_layer() const;
    /// Assign an existing native descriptor to the pipe end.
    /**
     * This function opens the descriptor to hold an existing native descriptor.
     *
     * @param native_descriptor A native descriptor.
     *
     * @throws asio::system_error Thrown on failure.
     */
    void assign(const native_handle_type& native_descriptor);

    /// Assign an existing native descriptor to the pipe end.
    /**
     * This function opens the descriptor to hold an existing native descriptor.
     *
     * @param native_descriptor A native descriptor.
     *
     * @param ec Set to indicate what error occurred, if any.
     */
    void assign(const native_handle_type& native_descriptor, asio::error_code& ec);

    /// The native representation of a descriptor.
    typedef implementation_defined native_handle_type;


    /// A descriptor is always the lowest layer.
    typedef basic_descriptor lowest_layer_type;
};


/// Provides pipe_write_end async communication functionality.
/**
 * The pipe_write_end class template provides asynchronous and
 * blocking stream-oriented pipe_write_end functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * Synchronous @c read_some and @c write_some operations are thread safe with
 * respect to each other, if the underlying operating system calls are also
 * thread safe. This means that it is permitted to perform concurrent calls to
 * these synchronous operations on a single pipe_write_end object. Other synchronous
 * operations, such as @c close, are not thread safe.
 *
 * @par Concepts:
 * AsyncReadStream, AsyncWriteStream, Stream, SyncReadStream, SyncWriteStream.
 */
template<typename Executor = any_io_executor>
struct basic_pipe_write_end
{
    /// The used stream type.
    typedef implementation_defined stream_type;

    /// The native representation of a descriptor.
    typedef implementation_defined native_handle_type;

    /// The type of the executor associated with the object.
    typedef Executor executor_type;


    /// Rebinds the pipe_write_end type to another executor.
    template <typename Executor1>
    struct rebind_executor
    {
        /// The pipe_write_end type when rebound to the specified executor.
        typedef basic_pipe_write_end<Executor1> other;
    };

    /// Construct a pipe_write_end and connect it.
    /**
     * This constructor creates a pipe_write_end and opens it.
     *
     * @param ex The I/O executor that the pipe_write_end will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe_write_end.
     */
    basic_pipe_write_end(Executor exec);

    /// Construct a pipe_write_end and connect it.
    /**
     * This constructor creates a pipe_write_end and opens it.
     *
     * @param ex The I/O executor that the pipe_write_end will use, by default, to
     * dispatch handlers for any asynchronous operations performed on the
     * pipe_write_end.
     */
    template <typename ExecutionContext>
    explicit basic_pipe_write_end(ExecutionContext& context);

    /// Move-construct a pipe_write_end from another.
    /**
     * This constructor moves a pipe_write_end from one object to another.
     *
     * @param other The other pipe_write_end object from which the move will occur.
     *
     * @note Following the move, the moved-from object is closed.
     */
    basic_pipe_write_end(basic_pipe_write_end && ) = default;
    /// Move-assign a pipe_write_end from another.
    /**
     * This assignment operator moves a pipe_write_end from one object to
     * another.
     *
     * @param other The other pipe_write_end object from which the move will occur.

     *
     * @note Following the move, the moved-from object is closed.
     */
    basic_pipe_write_end & operator=(basic_pipe_write_end && ) = default;

    /// Write some data to the pipe_write_end.
    /**
     * This function is used to write data to the pipe_write_end. The function
     * call will block until one or more bytes of the data has been written
     * successfully, or until an error occurs.
     *
     * @param buffers One or more data buffers to be written to the pipe_write_end.
     *
     * @returns The number of bytes written.
     *
     * @throws asio::system_error Thrown on failure. An error code of
     * asio::error::eof indicates that the connection was closed by the
     * peer.
     *
     * @note The write_some operation may not transmit all of the data to the
     * peer. Consider using the @ref write function if you need to ensure that
     * all data is written before the blocking operation completes.
     *
     * @par Example
     * To write a single data buffer use the @ref buffer function as follows:
     * @code
     * pipe_write_end.write_some(asio::buffer(data, size));
     * @endcode
     * See the @ref buffer documentation for information on writing multiple
     * buffers in one go, and how to use it with arrays, boost::array or
     * std::vector.
     */
    template <typename ConstBufferSequence>
    std::size_t write_some(const ConstBufferSequence& buffers);

    /// Write some data to the pipe_write_end.
    /**
     * This function is used to write data to the pipe_write_end. The function
     * call will block until one or more bytes of the data has been written
     * successfully, or until an error occurs.
     *
     * @param buffers One or more data buffers to be written to the pipe_write_end.
     *
     * @param ec Set to indicate what error occurred, if any.
     *
     * @returns The number of bytes written. Returns 0 if an error occurred.
     *
     * @note The write_some operation may not transmit all of the data to the
     * peer. Consider using the @ref write function if you need to ensure that
     * all data is written before the blocking operation completes.
     */
    template <typename ConstBufferSequence>
    std::size_t write_some(const ConstBufferSequence& buffers, asio::error_code& ec);

    /// Start an asynchronous write.
    /**
     * This function is used to asynchronously write data to the stream
     * pipe_write_end. The function call always returns immediately.
     *
     * @param buffers One or more data buffers to be written to the pipe_write_end.
     * Although the buffers object may be copied as necessary, ownership of the
     * underlying memory blocks is retained by the caller, which must guarantee
     * that they remain valid until the handler is called.
     *
     * @param handler The handler to be called when the write operation completes.
     * Copies will be made of the handler as required. The function signature of
     * the handler must be:
     * @code void handler(
     *   const asio::error_code& error, // Result of operation.
     *   std::size_t bytes_transferred           // Number of bytes written.
     * ); @endcode
     * Regardless of whether the asynchronous operation completes immediately or
     * not, the handler will not be invoked from within this function. On
     * immediate completion, invocation of the handler will be performed in a
     * manner equivalent to using asio::post().
     *
     * @note The write operation may not transmit all of the data to the peer.
     * Consider using the @ref async_write function if you need to ensure that all
     * data is written before the asynchronous operation completes.
     *
     * @par Example
     * To write a single data buffer use the @ref buffer function as follows:
     * @code
     * descriptor.async_write_some(asio::buffer(data, size), handler);
     * @endcode
     * See the @ref buffer documentation for information on writing multiple
     * buffers in one go, and how to use it with arrays, boost::array or
     * std::vector.
     */
    template <typename ConstBufferSequence,
            ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code, std::size_t)) WriteHandler
            ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
    ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler, void (asio::error_code, std::size_t))
    async_write_some(const ConstBufferSequence& buffers,
                     ASIO_MOVE_ARG(WriteHandler) handler  ASIO_DEFAULT_COMPLETION_TOKEN(executor_type));

    /// Close the pipe_write_end.
    /**
     * This function is used to close the pipe_write_end. Any asynchronous read or
     * write operations will be cancelled immediately, and will complete with the
     * asio::error::operation_aborted error.
     *
     * @throws asio::system_error Thrown on failure. Note that, even if
     * the function indicates an error, the underlying descriptor is closed.
     */
    void close();

    /// Close the pipe_write_end.
    /**
     * This function is used to close the pipe_write_end. Any asynchronous read or
     * write operations will be cancelled immediately, and will complete with the
     * asio::error::operation_aborted error.
     *
     * @param ec Set to indicate what error occurred, if any. Note that, even if
     * the function indicates an error, the underlying descriptor is closed.
     */
    void close(error_code & ec) noexcept;

    /// Cancel all asynchronous operations associated with the pipe_write_end.
    /**
     * This function causes all outstanding asynchronous read or write operations
     * to finish immediately, and the handlers for cancelled operations will be
     * passed the asio::error::operation_aborted error.
     *
     * @throws asio::system_error Thrown on failure.
     */
    void cancel();

    /// Cancel all asynchronous operations associated with the pipe_write_end.
    /**
     * This function causes all outstanding asynchronous read or write operations
     * to finish immediately, and the handlers for cancelled operations will be
     * passed the asio::error::operation_aborted error.
     *
     * @param ec Set to indicate what error occurred, if any.
     */
    void cancel(asio::error_code& ec);

    /// Determine whether the pipe_write_end is open.
    bool is_open() noexcept;


    /// Get the native pipe end representation.
    /**
     * This function may be used to obtain the underlying representation of the
     * descriptor. This is intended to allow access to native descriptor
     * functionality that is not otherwise provided.
     */
    native_handle_type native_handle() const;


    /// Get a reference to the lowest layer.
    /**
     * This function returns a reference to the lowest layer in a stack of
     * layers. Since a descriptor cannot contain any further layers, it
     * simply returns a reference to itself.
     *
     * @return A reference to the lowest layer in the stack of layers. Ownership
     * is not transferred to the caller.
     */
    lowest_layer_type& lowest_layer();

    /// Get a const reference to the lowest layer.
    /**
     * This function returns a const reference to the lowest layer in a stack of
     * layers. Since a descriptor cannot contain any further layers, it
     * simply returns a reference to itself.
     *
     * @return A const reference to the lowest layer in the stack of layers.
     * Ownership is not transferred to the caller.
     */
    const lowest_layer_type& lowest_layer() const;

    /// Assign an existing native descriptor to the pipe end.
    /**
     * This function opens the descriptor to hold an existing native descriptor.
     *
     * @param native_descriptor A native descriptor.
     *
     * @throws asio::system_error Thrown on failure.
     */
    void assign(const native_handle_type& native_descriptor);

    /// Assign an existing native descriptor to the pipe end.
    /**
     * This function opens the descriptor to hold an existing native descriptor.
     *
     * @param native_descriptor A native descriptor.
     *
     * @param ec Set to indicate what error occurred, if any.
     */
    void assign(const native_handle_type& native_descriptor, asio::error_code& ec);

    /// The native representation of a descriptor.
    typedef implementation_defined native_handle_type;


    /// A descriptor is always the lowest layer.
    typedef basic_descriptor lowest_layer_type;
};

#endif

/// Typedef for the typical usage of a pipe read end.
using pipe_read_end  = basic_pipe_read_end<>;
/// Typedef for the typical usage of a pipe write end.
using pipe_write_end = basic_pipe_write_end<>;
/// Typedef for the typical usage of a pipe.
using pipe = basic_pipe<>;


}

#endif //ASIO_BASIC_PIPE_HPP
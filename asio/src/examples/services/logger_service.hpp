#ifndef SERVICES_LOGGER_SERVICE_HPP
#define SERVICES_LOGGER_SERVICE_HPP

#include <asio.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <fstream>
#include <sstream>
#include <string>

namespace services {

/// Service implementation for the logger.
template <typename Allocator = std::allocator<void> >
class logger_service
  : private boost::noncopyable
{
public:
  /// The demuxer type for this service.
  typedef asio::basic_demuxer<
    asio::demuxer_service<Allocator> > demuxer_type;

  /// The backend implementation of a logger.
  struct logger_impl
  {
    explicit logger_impl(const std::string& id) : identifier(id) {}
    std::string identifier;
  };

  /// The type for an implementation of the logger.
  typedef logger_impl* impl_type;

  /// Constructor creates a thread to run a private demuxer.
  logger_service(demuxer_type& demuxer)
    : demuxer_(demuxer),
      work_demuxer_(),
      work_(new typename demuxer_type::work(work_demuxer_)),
      work_thread_(new asio::thread(
            boost::bind(&demuxer_type::run, &work_demuxer_)))
  {
  }

  /// Destructor shuts down the private demuxer.
  ~logger_service()
  {
    /// Indicate that we have finished with the private demuxer. Its
    /// demuxer::run() function will exit once all other work has completed.
    work_.reset();
    if (work_thread_)
      work_thread_->join();
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  /// Return a null logger implementation.
  impl_type null() const
  {
    return 0;
  }

  /// Create a new logger implementation.
  void create(impl_type& impl, const std::string& identifier)
  {
    impl = new logger_impl(identifier);
  }

  /// Destroy a logger implementation.
  void destroy(impl_type& impl)
  {
    delete impl;
    impl = null();
  }

  /// Set the output file for the logger. The current implementation sets the
  /// output file for all logger instances, and so the impl parameter is not
  /// actually needed. It is retained here to illustrate how service functions
  /// are typically defined.
  void use_file(impl_type& impl, const std::string& file)
  {
    // Pass the work of opening the file to the background thread.
    work_demuxer_.post(boost::bind(&logger_service::use_file_impl, this, file));
  }

  /// Log a message.
  void log(impl_type& impl, const std::string& message)
  {
    // Format the text to be logged.
    std::ostringstream os;
    os << boost::posix_time::microsec_clock::universal_time();
    os << " - " << impl->identifier << " - " << message;

    // Pass the work of opening the file to the background thread.
    work_demuxer_.post(boost::bind(&logger_service::log_impl, this, os.str()));
  }

private:
  /// Helper function used to open the output file from within the private
  /// demuxer's thread.
  void use_file_impl(const std::string& file)
  {
    ofstream_.close();
    ofstream_.clear();
    ofstream_.open(file.c_str());
  }

  /// Helper function used to log a message from within the private demuxer's
  /// thread.
  void log_impl(const std::string& text)
  {
    ofstream_ << text << std::endl;
  }

  /// The demuxer that owns this service.
  demuxer_type& demuxer_;

  /// Private demuxer used for performing logging operations.
  demuxer_type work_demuxer_;

  /// Work for the private demuxer to perform. If we do not give the demuxer
  /// some work to do then the demuxer::run() function will exit immediately.
  boost::scoped_ptr<typename demuxer_type::work> work_;

  /// Thread used for running the work demuxer's run loop.
  boost::scoped_ptr<asio::thread> work_thread_;

  /// The file to which log messages will be written.
  std::ofstream ofstream_;
};

} // namespace services

#endif // SERVICES_LOGGER_SERVICE_HPP

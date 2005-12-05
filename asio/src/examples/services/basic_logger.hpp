#ifndef SERVICES_BASIC_LOGGER_HPP
#define SERVICES_BASIC_LOGGER_HPP

#include <asio.hpp>
#include <boost/noncopyable.hpp>
#include <string>

namespace services {

/// Class to provide simple logging functionality. Use the services::logger
/// typedef.
template <typename Service>
class basic_logger
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide timer operations.
  typedef Service service_type;

  /// The native implementation type of the timer.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Constructor.
  /**
   * This constructor creates a logger.
   *
   * @param d The demuxer object used to locate the logger service.
   *
   * @param identifier An identifier for this logger.
   */
  explicit basic_logger(demuxer_type& d, const std::string& identifier)
    : service_(d.get_service(asio::service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_, identifier);
  }

  /// Destructor.
  ~basic_logger()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Set the output file for all logger instances.
  void use_file(const std::string& file)
  {
    service_.use_file(impl_, file);
  }

  /// Log a message.
  void log(const std::string& message)
  {
    service_.log(impl_, message);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace services

#endif // SERVICES_BASIC_LOGGER_HPP

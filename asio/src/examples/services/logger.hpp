#ifndef SERVICES_LOGGER_HPP
#define SERVICES_LOGGER_HPP

#include "basic_logger.hpp"
#include "logger_service.hpp"

namespace services {

/// Typedef for typical logger usage.
typedef basic_logger<logger_service> logger;

} // namespace services

#endif // SERVICES_LOGGER_HPP

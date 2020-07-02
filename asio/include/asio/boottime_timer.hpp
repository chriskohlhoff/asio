//
// boottime_timer.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2020 BSH Hausgeraete GmbH, Carl-Wery-Str. 34, 81739 Munich, Germany, www.bsh-group.de
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BOOTTIME_TIMER_HPP
#define ASIO_BOOTTIME_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_TIMERFD) \
  || defined(ASIO_HAS_STD_CHRONO) \
  || defined(GENERATING_DOCUMENTATION)

#include <chrono>
#include <ctime>
#include "asio/basic_waitable_timer.hpp"

namespace asio {


/**
 *
 * @brief A clock that uses the boottime clock which considers suspend time.
 *
 * It is very similar to std::chrono::steady_clock. Major customization is made in the static now function.
 * Here the standard C function clock_gettime is used with clock id CLOCK_BOOTTIME to get the right clock values.
 *
 */
struct boottime_clock
{
	typedef std::chrono::nanoseconds duration;
	typedef duration::rep rep;
	typedef duration::period period;
	typedef std::chrono::time_point<boottime_clock, duration> time_point;

	static constexpr bool is_steady = true;

	/**
	 *
	 * @brief Get the current value of the clock.
	 *
	 * clock_gettime with CLOCK_BOOTTIME is tried to be retrieved here. If this doesn't work std::chrono::steady_clock
	 * is used as fallback.
	 *
	 * @return The time_point representing the clock's current value.
	 *
	 */
	static time_point now() noexcept
	{
		struct timespec tp = {0, 0};
		time_point result;

		// Try to get boottime.
		if(clock_gettime(CLOCK_BOOTTIME, &tp) == 0)
		{
			result = time_point(std::chrono::seconds(tp.tv_sec) + std::chrono::nanoseconds(tp.tv_nsec));
		}
		else
		{
			// Use steady_clock as fallback.
			result = time_point(std::chrono::steady_clock::now().time_since_epoch());
		}
		return result;
	}
};

/// Typedef for a timer based on the boottime clock.
typedef basic_waitable_timer<boottime_clock> boottime_timer;

} // namespace asio

#endif // defined(ASIO_HAS_TIMERFD)
       //   || defined(ASIO_HAS_STD_CHRONO)
       //   || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_BOOTTIME_TIMER_HPP

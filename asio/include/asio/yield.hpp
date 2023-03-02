//
// yield.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2022 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "coroutine.hpp"

#if defined(__clang__)
# define GCC_SUPRESS_WARNING_FALLTROUGH _Pragma("GCC diagnostic push")\
										_Pragma("GCC diagnostic ignored \"-Wimplicit-fallthrough\"")
# define GCC_SUPRESS_WARNING_POP  _Pragma("GCC diagnostic pop")
#else
# define GCC_SUPRESS_WARNING_FALLTROUGH
# define GCC_SUPRESS_WARNING_POP
#endif

#ifndef reenter
# define reenter(c) GCC_SUPRESS_WARNING_FALLTROUGH \
					ASIO_CORO_REENTER(c) \
					GCC_SUPRESS_WARNING_POP
#endif

#ifndef yield
# define yield  GCC_SUPRESS_WARNING_FALLTROUGH \
				ASIO_CORO_YIELD \
			    GCC_SUPRESS_WARNING_POP
#endif

#ifndef fork
# define fork ASIO_CORO_FORK
#endif
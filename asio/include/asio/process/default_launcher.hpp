//
// process/default_launcher.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_PROCESS_DEFAULT_LAUNCHER_HPP
#define ASIO_PROCESS_DEFAULT_LAUNCHER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS)
#include "asio/process/windows/default_launcher.hpp"
#else
#if (defined(__MACH__) && defined(__APPLE__)) \
  || defined(__FreeBSD__) \
  || defined(__NetBSD__) \
  || defined(__OpenBSD__)
#include "asio/process/posix/pdfork_launcher.hpp"
#else
#include "asio/process/posix/default_launcher.hpp"
#endif
#endif

#include "asio/detail/push_options.hpp"

namespace asio
{

#if defined(ASIO_WINDOWS)
typedef windows::default_launcher default_process_launcher;
#else

#if (defined(__MACH__) && defined(__APPLE__)) \
  || defined(__FreeBSD__) \
  || defined(__NetBSD__) \
  || defined(__OpenBSD__)
typedef posix::pdfork_launcher default_process_launcher;
#else
typedef posix::default_launcher default_process_launcher;
#endif
#endif
}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_DEFAULT_LAUNCHER_HPP

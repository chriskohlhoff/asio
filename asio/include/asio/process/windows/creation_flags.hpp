//
// process/environment.hpp
// ~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#ifndef ASIO_PROCESS_WINDOWS_CREATION_FLAGS_HPP
#define ASIO_PROCESS_WINDOWS_CREATION_FLAGS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/process/default_launcher.hpp"
#include "asio/detail/push_options.hpp"

namespace asio
{
namespace windows
{

template<DWORD Flags>
struct process_creation_flags
{
  error_code on_setup(windows::default_launcher & launcher, const filesystem::path &, const std::wstring &) const
  {
    launcher.startup_info.StartupInfo.dwFlags |= Flags;
    return error_code {};
  };
};

ASIO_CONSTEXPR static process_creation_flags<CREATE_NEW_PROCESS_GROUP> create_new_process_group;

}
}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_WINDOWS_CREATION_FLAGS_HPP

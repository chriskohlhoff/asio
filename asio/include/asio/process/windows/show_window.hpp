//
// process/environment.hpp
// ~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#ifndef ASIO_PROCESS_WINDOWS_SHOW_WINDOWS_HPP
#define ASIO_PROCESS_WINDOWS_SHOW_WINDOWS_HPP

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
struct process_show_window
{
  error_code on_setup(windows::default_launcher & launcher, const filesystem::path &, const std::wstring &) const
  {
    launcher.startup_info.StartupInfo.dwFlags |= STARTF_USESHOWWINDOW;
    launcher.startup_info.StartupInfo.wShowWindow |= Flags;

    return error_code {};
  };
};

///Hides the window and activates another window.
ASIO_CONSTEXPR static process_show_window<SW_HIDE           > show_window_hide{};
///Activates the window and displays it as a maximized window.
ASIO_CONSTEXPR static process_show_window<SW_SHOWMAXIMIZED  > show_window_maximized{};
///Activates the window and displays it as a minimized window.
ASIO_CONSTEXPR static process_show_window<SW_SHOWMINIMIZED  > show_window_minimized{};
///Displays the window as a minimized window. This value is similar to `minimized`, except the window is not activated.
ASIO_CONSTEXPR static process_show_window<SW_SHOWMINNOACTIVE> show_window_minimized_not_active{};
///Displays a window in its most recent size and position. This value is similar to show_normal`, except that the window is not activated.
ASIO_CONSTEXPR static process_show_window<SW_SHOWNOACTIVATE > show_window_not_active{};
///Activates and displays a window. If the window is minimized or maximized, the system restores it to its original size and position. An application should specify this flag when displaying the window for the first time.
ASIO_CONSTEXPR static process_show_window<SW_SHOWNORMAL     > show_window_normal{};

}
}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_PROCESS_WINDOWS_SHOW_WINDOWS_HPP

//
// push_options.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

// No header guard

#if defined (_MSC_VER)
# pragma warning (disable:4103)
# pragma warning (push)
# pragma warning (disable:4355)
# pragma pack (push, 8)
// Note that if the /Og optimisation flag is enabled with MSVC6, the compiler
// has a tendency to incorrectly optimise away some calls to member template
// functions, even though those functions contain code that should not be
// optimised away! Therefore we will always disable this optimisation option
// for the MSVC6 compiler.
# if (_MSC_VER < 1300)
#  pragma optimize ("g", off)
# endif
#elif defined (__BORLANDC__)
# pragma option push -a8 -b -Ve- -Vx-
# pragma nopushoptwarn
# pragma nopackwarning
#elif defined (__MINGW32__)
# pragma pack (push, 8)
#endif

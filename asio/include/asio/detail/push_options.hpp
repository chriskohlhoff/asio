//
// push_options.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
# pragma option push -a8 -b -Ve- -Vx- -w-inl
# pragma nopushoptwarn
# pragma nopackwarning
#elif defined (__MINGW32__)
# pragma pack (push, 8)
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// define a helper macro "D" to append to the ostream. only enabled in debug build

#ifdef D
#undef D
#endif

#ifndef NDEBUG  // if debug build
#define D(str) << str
#else
#define D(str)
#endif

// define a helper macro "DSS" to append to the ostream. only enabled in debug build

#ifdef DSS
#undef DSS
#endif

#ifndef NDEBUG  // if debug build
#define DSS ss
#else
#define DSS if constexpr (false) ss
#endif

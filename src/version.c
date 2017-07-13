//   Copyright 2017 Vidrio Technologies
//   by Nathan Clack <nathan@vidriotech.com>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

const char * heffalump_api_version() {    
    return STR(GIT_TAG) "-" STR(GIT_HASH);    
}
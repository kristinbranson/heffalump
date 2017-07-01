#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)

const char * heffalump_api_version() {    
    return STR(GIT_TAG) "-" STR(GIT_HASH);    
}
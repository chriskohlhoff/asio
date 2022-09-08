inline hostent* gethostbyaddr_orbis(const char* addr, int length, int af, hostent* result, char* buffer, int buflength, asio::error_code& ec)
{
    static struct hostent* he = []() -> hostent*
    {
        static struct hostent he;
        // One time allocation for the name.
        he.h_name = new char[SCE_NET_RESOLVER_HOSTNAME_LEN_MAX + 1];
        return &he;
    } ();

    static char *aliases[1] = { nullptr };
    static char *addr_list[2] = { nullptr, nullptr };
    static struct SceNetInAddr sce_addr;

    memset(he->h_name, 0, SCE_NET_RESOLVER_HOSTNAME_LEN_MAX + 1);

    const int pool = sceNetPoolCreate("gethostbyaddr pool", 4 * 1024, 0);
    if(pool >= 0)
    {
        const SceNetId resolver = sceNetResolverCreate("gethostbyaddr resolver", pool, 0);
        if(resolver >= 0)
        {
            if(sceNetInetPton(SCE_NET_AF_INET, addr, &sce_addr.s_addr) > 0)
            {
                if(sceNetResolverStartAton(resolver, &sce_addr, he->h_name, SCE_NET_RESOLVER_HOSTNAME_LEN_MAX + 1, 0, 0, 0) >= 0)
                {
                    he->h_aliases = aliases;
                    he->h_length = 4;
                    he->h_addr_list = addr_list;
                    he->h_addrtype = AF_INET;
                }
            }
        }
        sceNetResolverDestroy(resolver);
    }
    sceNetPoolDestroy(pool);

    return he;
}

inline hostent* gethostbyname_orbis(const char* name, int af, struct hostent* result, char* buffer, int buflength, int ai_flags, asio::error_code& ec)
{
    static struct hostent he;
    static char *aliases[1] = { nullptr };
    static char *addr_list[2] = { nullptr, nullptr };
    static struct SceNetInAddr addr;

    struct hostent* return_value = nullptr;

    const int pool = sceNetPoolCreate("gethostbyname pool", 4 * 1024, 0);
    if(pool >= 0)
    {
        const SceNetId resolver = sceNetResolverCreate("gethostbyname resolver", pool, 0);
        if(resolver >= 0)
        {
            const int ntoa_result = sceNetResolverStartNtoa(resolver, name, &addr, 0, 0, 0);
            if(ntoa_result >= 0)
            {
                addr_list[0] = (char *)&addr;

                he.h_name = (char *)name;
                he.h_aliases = aliases;
                he.h_length = 4;
                he.h_addr_list = addr_list;
                he.h_addrtype = AF_INET;

                return_value = &he;
            }
        }
        sceNetResolverDestroy(resolver);
    }
    sceNetPoolDestroy(pool);

    return return_value;
}


#include "utils.h"

#include <unistd.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#if defined(__APPLE__)
#include <mach/task.h>
#include <mach/mach_init.h>
#endif

namespace nalu {

size_t current_memory_usage()
{
    size_t vmsize = 0;

#ifdef __linux__
    std::ifstream statm("/proc/self/statm");

    if (statm) {
        size_t vm_pages;
        size_t rss_pages;
        statm >> vm_pages >> rss_pages;

        vmsize = rss_pages * sysconf(_SC_PAGESIZE);
    }

#elif defined(__APPLE__)
    // From trilinos/packages/stk/stk_util/stk_util/environment/memory_util.cpp
    struct task_basic_info tinfo;
    mach_msg_type_number_t tinfo_count = TASK_BASIC_INFO_COUNT;

    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO,
                                  reinterpret_cast<task_info_t>(&tinfo),
                                  &tinfo_count)) {
        return 0;
    }
    vmsize = tinfo.resident_size;
#endif

    return vmsize;
}

std::string human_bytes(const size_t bytes)
{
    std::ostringstream hbytes;
    const double kb = 1024.0;
    const double mb = kb * kb;
    const double gb = mb * kb;

    const double tmp = static_cast<double>(bytes);

    if (tmp >= gb)
        hbytes << std::setw(8) << std::fixed << std::setprecision(3)
               << tmp / gb << " GB";
    else if (tmp >= mb)
        hbytes << std::setw(8) << std::fixed << std::setprecision(3)
               << tmp / mb << " MB";
    else if (tmp >= kb)
        hbytes << std::setw(8) << std::fixed << std::setprecision(3)
               << tmp / kb << " KB";
    else
        hbytes << std::setw(8) << std::fixed << std::setprecision(3)
               << tmp << "  B";

    return hbytes.str();
}

} // nalu

#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <string>

namespace nalu {

size_t current_memory_usage();

std::string human_bytes(const size_t);

} // nalu

#endif /* UTILS_H */

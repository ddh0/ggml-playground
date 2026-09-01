/* logger.cpp */

#include "logger.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace ANSI;

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const time_t now_t = std::chrono::system_clock::to_time_t(now);
    const tm now_tm = *std::localtime(&now_t);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%F %a %T") << '.' << std::setfill('0')
        << std::setw(4) << ((us.count() / 100) % 10000);
    return oss.str();
}

// init
logger::logger(
    std::FILE * dbg_stream,
    std::FILE * inf_stream,
    std::FILE * wrn_stream,
    std::FILE * err_stream
):
    dbg_stream_(dbg_stream),
    inf_stream_(inf_stream),
    wrn_stream_(wrn_stream),
    err_stream_(err_stream) {}

// get prefix string
std::string logger::prefix(log_lvl lvl) const {
    std::ostringstream oss; oss << mode_bold_set;
    switch (lvl) {
        case log_lvl::debug:   oss << fg_bright_white  << "  DEBUG"; break;
        case log_lvl::info:    oss << fg_bright_green  << "   INFO"; break;
        case log_lvl::warning: oss << fg_bright_yellow << "WARNING"; break;
        case log_lvl::error:   oss << fg_bright_red    << "  ERROR"; break;
        default: throw std::runtime_error("invalid log_lvl");
    }
    oss << mode_reset_all << mode_bold_set << fg_bright_black << ":" << mode_reset_all;
    return oss.str();
}

// get full log string w/ prefix and styling
std::string logger::get_str(log_lvl lvl, std::string_view msg) const {
    std::ostringstream oss;
    oss << mode_bold_set << fg_bright_black << "[" << timestamp() << "]"
        << mode_reset_all << " " << prefix(lvl) << " " << msg;
    return oss.str();
}

// manually specify a log level and message
void logger::operator()(log_lvl lvl, std::string_view msg) const {
    switch (lvl) {
        case log_lvl::debug:   std::fprintf(dbg_stream_, "%s\n", get_str(lvl, msg).c_str()); break;
        case log_lvl::info:    std::fprintf(inf_stream_, "%s\n", get_str(lvl, msg).c_str()); break;
        case log_lvl::warning: std::fprintf(wrn_stream_, "%s\n", get_str(lvl, msg).c_str()); break;
        case log_lvl::error:   std::fprintf(err_stream_, "%s\n", get_str(lvl, msg).c_str()); break;
        default: throw std::runtime_error("unreachable");
    }
}

// print debug message to stdout
void logger::debug(std::string_view msg) const { (*this)(log_lvl::debug,   msg); }
// print info message to stdout
void logger::info (std::string_view msg) const { (*this)(log_lvl::info,    msg); }
// print warning message to stderr
void logger::warn (std::string_view msg) const { (*this)(log_lvl::warning, msg); }
// print error message to stderr
void logger::error(std::string_view msg) const { (*this)(log_lvl::error,   msg); }

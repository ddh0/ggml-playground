/* logger.cpp */
#include "logger.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace ANSI;

std::string timestamp() {
    const auto now   = std::chrono::system_clock::now();
    const auto timer = std::chrono::system_clock::to_time_t(now);
    const auto tm    = *std::localtime(&timer);
    const auto ms    = std::chrono::duration_cast<std::chrono::milliseconds>
        (now.time_since_epoch()) % 1000;
    std::ostringstream oss;
    oss << std::put_time(&tm, "%F %T") << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}


logger::logger(std::FILE * stdout_, std::FILE * stderr_): out_(stdout_), err_(stderr_) {}

int logger::clamp_lvl(int lvl) const noexcept {
    const int lo = static_cast<int>(log_lvl::debug), hi = static_cast<int>(log_lvl::error);
    if (lvl < lo) return lo; if (lvl > hi) return hi;
    return lvl;
}

std::string logger::prefix(int lvl) {
    std::ostringstream oss;
    oss << mode_bold_set;
    switch (lvl) {
        case 0:  oss << fg_bright_white  << "  DEBUG"; break;
        case 1:  oss << fg_bright_green  << "   INFO"; break;
        case 2:  oss << fg_bright_yellow << "WARNING"; break;
        case 3:  oss << fg_bright_red    << "  ERROR"; break;
        default: oss << fg_bright_cyan   << "???????";
    }
    oss << mode_reset_all << mode_bold_set << fg_bright_black << ":" << mode_reset_all;
    return oss.str();
}

std::string logger::get_str(log_lvl lvl, std::string_view msg) const {
    std::ostringstream oss;
    oss << mode_bold_set << fg_bright_black << "[" << timestamp() << "]" << mode_reset_all
    << " " << prefix(clamp_lvl(static_cast<int>(lvl))) << " " << msg;
    return oss.str();
}

void logger::operator()(int level, std::string_view msg) {
    auto const clamped = clamp_lvl(level);
    auto *     stream  = (clamped >= static_cast<int>(log_lvl::warning)) ? err_ : out_;

    std::ostringstream oss;
    oss << mode_bold_set << fg_bright_black << "[" << timestamp() << "]"
        << mode_reset_all << " " << prefix(clamped) << " " << msg;

    std::fprintf(stream, "%s\n", oss.str().c_str());
}

void logger::operator()(log_lvl level, std::string_view msg) {
    (*this)(static_cast<int>(level), msg);
}

void logger::debug(std::string_view msg) { (*this)(log_lvl::debug,   msg); }
void logger::info (std::string_view msg) { (*this)(log_lvl::info,    msg); }
void logger::warn (std::string_view msg) { (*this)(log_lvl::warning, msg); }
void logger::error(std::string_view msg) { (*this)(log_lvl::error,   msg); }

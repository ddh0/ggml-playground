/* logger.h */
#pragma once

#include <cstdio>
#include <string>
#include <string_view>

// ANSI codes for terminal emulators
namespace ANSI {

    // standard foreground colors
    constexpr inline const char * fg_black   = "\033[30m";
    constexpr inline const char * fg_red     = "\033[31m";
    constexpr inline const char * fg_green   = "\033[32m";
    constexpr inline const char * fg_yellow  = "\033[33m";
    constexpr inline const char * fg_blue    = "\033[34m";
    constexpr inline const char * fg_magenta = "\033[35m";
    constexpr inline const char * fg_cyan    = "\033[36m";
    constexpr inline const char * fg_white   = "\033[37m";

    // standard background colors
    constexpr inline const char * bg_black   = "\033[40m";
    constexpr inline const char * bg_red     = "\033[41m";
    constexpr inline const char * bg_green   = "\033[42m";
    constexpr inline const char * bg_yellow  = "\033[43m";
    constexpr inline const char * bg_blue    = "\033[44m";
    constexpr inline const char * bg_magenta = "\033[45m";
    constexpr inline const char * bg_cyan    = "\033[46m";
    constexpr inline const char * bg_white   = "\033[47m";

    // bright foreground colors
    constexpr inline const char * fg_bright_black   = "\033[90m";
    constexpr inline const char * fg_bright_red     = "\033[91m";
    constexpr inline const char * fg_bright_green   = "\033[92m";
    constexpr inline const char * fg_bright_yellow  = "\033[93m";
    constexpr inline const char * fg_bright_blue    = "\033[94m";
    constexpr inline const char * fg_bright_magenta = "\033[95m";
    constexpr inline const char * fg_bright_cyan    = "\033[96m";
    constexpr inline const char * fg_bright_white   = "\033[97m";

    // bright background colors
    constexpr inline const char * bg_bright_black   = "\033[100m";
    constexpr inline const char * bg_bright_red     = "\033[101m";
    constexpr inline const char * bg_bright_green   = "\033[102m";
    constexpr inline const char * bg_bright_yellow  = "\033[103m";
    constexpr inline const char * bg_bright_blue    = "\033[104m";
    constexpr inline const char * bg_bright_magenta = "\033[105m";
    constexpr inline const char * bg_bright_cyan    = "\033[106m";
    constexpr inline const char * bg_bright_white   = "\033[107m";

    // text modes
    constexpr inline const char * mode_reset_all           = "\033[0m";
    constexpr inline const char * mode_bold_set            = "\033[1m";
    constexpr inline const char * mode_dim_set             = "\033[2m";
    constexpr inline const char * mode_italic_set          = "\033[3m";
    constexpr inline const char * mode_underline_set       = "\033[4m";
    constexpr inline const char * mode_blinking_set        = "\033[5m";
    constexpr inline const char * mode_reverse_set         = "\033[7m";
    constexpr inline const char * mode_hidden_set          = "\033[8m";
    constexpr inline const char * mode_strikethrough_set   = "\033[9m";
    constexpr inline const char * mode_bold_reset          = "\033[22m";
    constexpr inline const char * mode_dim_reset           = "\033[22m";
    constexpr inline const char * mode_italic_reset        = "\033[23m";
    constexpr inline const char * mode_underline_reset     = "\033[24m";
    constexpr inline const char * mode_blinking_reset      = "\033[25m";
    constexpr inline const char * mode_reverse_reset       = "\033[27m";
    constexpr inline const char * mode_hidden_reset        = "\033[28m";
    constexpr inline const char * mode_strikethrough_reset = "\033[29m";

    // special
    constexpr inline const char * bell             = "\a";
    constexpr inline const char * terminal_reset   = "\033""c";
    constexpr inline const char * scrollback_clear = "\033[3J";

    // reset terminal + clear scrollback buffer + reset text modes
    constexpr inline const char * clear = "\033""c\033[3J\033[0m";

} // namespace ANSI

// get string timestamp for current time
std::string timestamp();

enum class log_lvl : int {
    debug   = 0,
    info    = 1,
    warning = 2,
    error   = 3
};

// utility to print log messages with ANSI styling
struct logger {

    // by default, initialize with stdout for `debug` and `info` and stderr for `warn` and `error`
    logger(
        std::FILE * dbg_stream_ = stdout,
        std::FILE * inf_stream_ = stdout,
        std::FILE * wrn_stream_ = stderr,
        std::FILE * err_stream_ = stderr
    );

    void operator()(log_lvl level, std::string_view msg) const;

    void debug(std::string_view msg) const;
    void info (std::string_view msg) const;
    void warn (std::string_view msg) const;
    void error(std::string_view msg) const;

    std::string get_str(log_lvl lvl, std::string_view msg) const;

    // printf-style formatting support
    template <typename... Args>
    std::string format(std::string_view fmt, const Args&... args) const {
        int size = std::snprintf(nullptr, 0, fmt.data(), args...);
        std::string buf(size, '\0');
        std::snprintf(buf.data(), size + 1, fmt.data(), args...);
        return buf;
    }

    template <typename... Args>
    void operator()(log_lvl lvl, std::string_view fmt, const Args&... args) const {
        (*this)(lvl, format(fmt, args...));
    }

    template <typename... Args>
    void debug(std::string_view fmt, const Args&... args) const {
        (*this)(log_lvl::debug,   fmt, args...);
    }
    template <typename... Args>
    void info (std::string_view fmt, const Args&... args) const {
        (*this)(log_lvl::info,    fmt, args...);
    }
    template <typename... Args>
    void warn (std::string_view fmt, const Args&... args) const {
        (*this)(log_lvl::warning, fmt, args...);
    }
    template <typename... Args>
    void error(std::string_view fmt, const Args&... args) const {
        (*this)(log_lvl::error,   fmt, args...);
    }

    private:
        std::FILE * dbg_stream_;
        std::FILE * inf_stream_;
        std::FILE * wrn_stream_;
        std::FILE * err_stream_;

        std::string prefix(log_lvl lvl) const;
};

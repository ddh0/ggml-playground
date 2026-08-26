/* logger.h */
#pragma once

#include <cstdio>
#include <string>
#include <string_view>

// ANSI codes for terminal emulators
namespace ANSI {

    // standard foreground colors
    constexpr inline const char * fg_black   = "\x1b[30m";
    constexpr inline const char * fg_red     = "\x1b[31m";
    constexpr inline const char * fg_green   = "\x1b[32m";
    constexpr inline const char * fg_yellow  = "\x1b[33m";
    constexpr inline const char * fg_blue    = "\x1b[34m";
    constexpr inline const char * fg_magenta = "\x1b[35m";
    constexpr inline const char * fg_cyan    = "\x1b[36m";
    constexpr inline const char * fg_white   = "\x1b[37m";

    // standard background colors
    constexpr inline const char * bg_black   = "\x1b[40m";
    constexpr inline const char * bg_red     = "\x1b[41m";
    constexpr inline const char * bg_green   = "\x1b[42m";
    constexpr inline const char * bg_yellow  = "\x1b[43m";
    constexpr inline const char * bg_blue    = "\x1b[44m";
    constexpr inline const char * bg_magenta = "\x1b[45m";
    constexpr inline const char * bg_cyan    = "\x1b[46m";
    constexpr inline const char * bg_white   = "\x1b[47m";

    // bright foreground colors
    constexpr inline const char * fg_bright_black   = "\x1b[90m";
    constexpr inline const char * fg_bright_red     = "\x1b[91m";
    constexpr inline const char * fg_bright_green   = "\x1b[92m";
    constexpr inline const char * fg_bright_yellow  = "\x1b[93m";
    constexpr inline const char * fg_bright_blue    = "\x1b[94m";
    constexpr inline const char * fg_bright_magenta = "\x1b[95m";
    constexpr inline const char * fg_bright_cyan    = "\x1b[96m";
    constexpr inline const char * fg_bright_white   = "\x1b[97m";

    // bright background colors
    constexpr inline const char * bg_bright_black   = "\x1b[100m";
    constexpr inline const char * bg_bright_red     = "\x1b[101m";
    constexpr inline const char * bg_bright_green   = "\x1b[102m";
    constexpr inline const char * bg_bright_yellow  = "\x1b[103m";
    constexpr inline const char * bg_bright_blue    = "\x1b[104m";
    constexpr inline const char * bg_bright_magenta = "\x1b[105m";
    constexpr inline const char * bg_bright_cyan    = "\x1b[106m";
    constexpr inline const char * bg_bright_white   = "\x1b[107m";

    // text modes
    constexpr inline const char * mode_reset_all           = "\x1b[0m";
    constexpr inline const char * mode_bold_set            = "\x1b[1m";
    constexpr inline const char * mode_dim_set             = "\x1b[2m";
    constexpr inline const char * mode_italic_set          = "\x1b[3m";
    constexpr inline const char * mode_underline_set       = "\x1b[4m";
    constexpr inline const char * mode_blinking_set        = "\x1b[5m";
    constexpr inline const char * mode_reverse_set         = "\x1b[7m";
    constexpr inline const char * mode_hidden_set          = "\x1b[8m";
    constexpr inline const char * mode_strikethrough_set   = "\x1b[9m";
    constexpr inline const char * mode_bold_reset          = "\x1b[22m";
    constexpr inline const char * mode_dim_reset           = "\x1b[22m";
    constexpr inline const char * mode_italic_reset        = "\x1b[23m";
    constexpr inline const char * mode_underline_reset     = "\x1b[24m";
    constexpr inline const char * mode_blinking_reset      = "\x1b[25m";
    constexpr inline const char * mode_reverse_reset       = "\x1b[27m";
    constexpr inline const char * mode_hidden_reset        = "\x1b[28m";
    constexpr inline const char * mode_strikethrough_reset = "\x1b[29m";

    // special
    constexpr inline const char * bell             = "\a";
    constexpr inline const char * terminal_reset   = "\x1b""c";
    constexpr inline const char * scrollback_clear = "\x1b[3J";

    // reset terminal + clear scrollback buffer + reset all text modes
    constexpr inline const char * clear = "\x1b""c\x1b[3J\x1b[0m";

} // namespace ANSI

std::string timestamp();

enum class log_lvl : int {
    debug   = 0,
    info    = 1,
    warning = 2,
    error   = 3
};

struct logger {
    public:
        explicit logger(std::FILE * stdout_ = stdout, std::FILE * stderr_ = stderr);

        void operator()(int     level, std::string_view msg);
        void operator()(log_lvl level, std::string_view msg);

        void debug(std::string_view msg);
        void info (std::string_view msg);
        void warn (std::string_view msg);
        void error(std::string_view msg);

        std::string get_str(log_lvl lvl, std::string_view msg) const;

    private:
        std::FILE * out_;
        std::FILE * err_;

        static std::string prefix(int level);
        int clamp_lvl(int level) const noexcept;
};

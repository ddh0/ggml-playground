/* ggml-playground.cpp */

#include "logger.h"
#include "ggml.h"

int main() {
    logger log;
    for (int i = 0; i < 128; ++i) {
        log.info("i == %d", i);
    }
    return EXIT_SUCCESS;
}

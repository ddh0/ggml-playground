/* ggml-playground.cpp */

#include "logger.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"

#include <cstdlib>
#include <cmath>

static constexpr int N_THREADS = 8;
static constexpr float EPS = 1e-5f;

int main() {
    logger log;
    log.info("start");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = true // tensor data is allocated in the backend
    };

    struct ggml_context * ctx = ggml_init(params);
    log.debug("ggml context init: mem_size = %zu", params.mem_size);

    ggml_backend_t backend = ggml_backend_metal_init();
    if (backend == NULL) {
        log.warn("could not init Metal backend, falling back to CPU");
        backend = ggml_backend_cpu_init();
    }

    log.info("backend init: name = %s, device = %s",
        ggml_backend_name(backend), ggml_backend_dev_name(ggml_backend_get_device(backend)));

    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        log.info("CPU: set n_threads = %d", N_THREADS);
        ggml_backend_cpu_set_n_threads(backend, N_THREADS);
    }

    // define the input scalar, label it, and mark it as input for the scheduler
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(x, "x"); ggml_set_input(x);

    // define and label the scalar parameters
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); ggml_set_name(a, "a");
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); ggml_set_name(b, "b");

    // define and label x^2 intermediate node
    struct ggml_tensor * x2 = ggml_mul(ctx, x, x); ggml_set_name(x2, "x^2");

    // define the function, label it, and mark it as output for the scheduler
    struct ggml_tensor * f = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
    ggml_set_name(f, "f"); ggml_set_output(f);

    // allocate all tensors in the backend buffer
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    log.debug("allocated backend buffer: size = %zu", ggml_backend_buffer_get_size(buf));

    struct ggml_cgraph * gf = ggml_new_graph(ctx); // initialize a computation graph from the ctx
    ggml_build_forward_expand(gf, f);              // build and expand the function graph forward
    log.debug("forward graph built: %d nodes", ggml_graph_n_nodes(gf));

    // set input tensors for the function
    const float x_val = 2.0f; ggml_backend_tensor_set(x, &x_val, 0, sizeof(float));
    const float a_val = 3.0f; ggml_backend_tensor_set(a, &a_val, 0, sizeof(float));
    const float b_val = 4.0f; ggml_backend_tensor_set(b, &b_val, 0, sizeof(float));
    log.info("set inputs: x = %.4f, a = %.4f, b = %.4f", x_val, a_val, b_val);

    float f_val = 0.0f;                                   // initialize the return value
    ggml_backend_graph_compute(backend, gf);              // compute the graph forward
    ggml_backend_tensor_get(f, &f_val, 0, sizeof(float)); // store the return value
    log.info("f(x) = a*x^2 + b = %.4f", f_val);

    const float expected = (a_val * x_val * x_val) + b_val;

    if (std::fabs(f_val - expected) > EPS) {
        log.error("unexpected forward computation result: got %.4f, expected %.4f", f_val, expected);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return EXIT_FAILURE;
    } else {
        log.info("forward computation result matches expected value: %.4f", expected);
    }

    // re-compute with a new input value
    const float new_x = -9.1055f;
    ggml_backend_tensor_set(x, &new_x, 0, sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(f, &f_val, 0, sizeof(float));
    log.info("re-ran graph with x = %.4f --> f(x) = %.4f", new_x, f_val);

    log.info("cleanup...");
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    log.info("done");

    return EXIT_SUCCESS;
}

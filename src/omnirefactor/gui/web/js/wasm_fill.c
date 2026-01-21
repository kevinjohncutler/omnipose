#include <stdint.h>

static uint32_t g_run_count = 0;
static uint32_t g_border_count = 0;
static uint32_t g_run_offset = 0;
static uint32_t g_border_offset = 0;
static uint32_t g_width = 0;
static uint32_t g_height = 0;
static uint32_t g_plane_size = 0;
static uint32_t g_steps_offset = 0;
static uint32_t g_step_count = 0;
static uint32_t g_mask_offset = 0;

__attribute__((export_name("fill_indices")))
void fill_indices(uint32_t mask_offset,
                  uint32_t indices_offset,
                  uint32_t count,
                  uint32_t value) {
    uint32_t *mask = (uint32_t *)(uintptr_t)mask_offset;
    const uint32_t *indices = (const uint32_t *)(uintptr_t)indices_offset;
    for (uint32_t i = 0; i < count; ++i) {
        mask[indices[i]] = value;
    }
}

__attribute__((export_name("setup_fill_runs")))
void setup_fill_runs(uint32_t mask_offset,
                     uint32_t width,
                     uint32_t height,
                     uint32_t run_table_offset,
                     uint32_t border_table_offset,
                     uint32_t steps_offset,
                     uint32_t step_count)
{
    g_mask_offset = mask_offset;
    g_width = width;
    g_height = height;
    g_plane_size = width * height;
    g_run_offset = run_table_offset;
    g_border_offset = border_table_offset;
    g_steps_offset = steps_offset;
    g_step_count = step_count;
}

__attribute__((export_name("reset_fill_runs")))
void reset_fill_runs()
{
    g_run_count = 0;
    g_border_count = 0;
}

__attribute__((export_name("emit_fill_run")))
void emit_fill_run(uint32_t start_idx, uint32_t end_idx, uint32_t prev_label, uint32_t new_label)
{
    uint32_t *mask = (uint32_t *)(uintptr_t)g_mask_offset;
    uint32_t *run_table = (uint32_t *)(uintptr_t)g_run_offset;
    const int32_t *steps = (const int32_t *)(uintptr_t)g_steps_offset;

    uint32_t write_pos = g_run_count * 4;
    run_table[write_pos + 0] = start_idx;
    run_table[write_pos + 1] = end_idx;
    run_table[write_pos + 2] = prev_label;
    run_table[write_pos + 3] = new_label;
    g_run_count += 1;

    for (uint32_t idx = start_idx; idx < end_idx; ++idx) {
        mask[idx] = new_label;
        const uint32_t x = idx % g_width;
        const uint32_t y = idx / g_width;
        for (uint32_t s = 0; s < g_step_count; ++s) {
            const int32_t dy = steps[2 * s + 0];
            const int32_t dx = steps[2 * s + 1];
            const int32_t nx = (int32_t)x + dx;
            const int32_t ny = (int32_t)y + dy;
            if (nx < 0 || nx >= (int32_t)g_width || ny < 0 || ny >= (int32_t)g_height) {
                continue;
            }
            const uint32_t nidx = (uint32_t)ny * g_width + (uint32_t)nx;
            if ((mask[nidx] | 0u) != new_label) {
                uint32_t slot = g_border_count * 2;
                uint32_t *border_table = (uint32_t *)(uintptr_t)g_border_offset;
                border_table[slot + 0] = idx;
                border_table[slot + 1] = g_step_count;
                g_border_count += 1;
                break;
            }
        }
    }
}

__attribute__((export_name("get_run_count")))
uint32_t get_run_count()
{
    return g_run_count;
}

__attribute__((export_name("get_border_count")))
uint32_t get_border_count()
{
    return g_border_count;
}

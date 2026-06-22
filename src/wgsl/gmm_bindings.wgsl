#define_import_path stt::gmm_bindings

// for log_gmm
@group(0) @binding(0) var<storage, read_write> gmm_mean: array<f32>;
@group(0) @binding(1) var<storage, read_write> gmm_covar: array<f32>;
@group(0) @binding(2) var<storage, read_write> gmm_weight: array<f32>;
@group(0) @binding(3) var<storage, read> observations: array<f32>;
@group(0) @binding(4) var<storage, read_write> tmp_result: array<f32>;
@group(0) @binding(5) var<storage, read_write> result: array<f32>;
@group(0) @binding(6) var<storage, read> state_id: array<u32,1>;

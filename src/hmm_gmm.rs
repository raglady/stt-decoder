pub mod gaussian;
pub mod gmm;

#[cfg(feature = "wgpu")]
use std::{borrow::Cow, collections::HashMap};

#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "wgpu")]
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};

#[cfg(feature = "wgpu")]
use wgpu::{PipelineCompilationOptions, util::DeviceExt};

use ndarray::{
    ArcArray2, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis,
};
#[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use crate::gpu::{CUDA_CONTEXT, CUDA_MODULE};

#[cfg(feature = "wgpu")]
use crate::gpu::{get_gpu_device_and_queue, storage_entry};

use crate::{hmm_gmm::gmm::Gmm, log_sum_exp, types::Float};

// ============= Hidden Markov Model =============

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HMMGMM {
    n_states: usize,
    state: Vec<Gmm>,
    log_transition: Array2<Float>,
    log_initial: Array1<Float>,
    log_final: Array1<Float>,
    variance_floor: Array1<Float>,
}

impl HMMGMM {
    pub fn new(n_states: usize, n_components: usize, dim: usize) -> Self {
        let mut log_transition =
            Array2::<Float>::from_elem([n_states, n_states], Float::NEG_INFINITY);
        cfg_select! {

            feature = "f64" => {
                for i in 0..n_states {
                    if i < n_states - 1 {
                        log_transition[[i, i]] = 0.5f64.ln();
                        log_transition[[i, i + 1]] = 0.5f64.ln();
                    } else {
                        log_transition[[i, i]] = 0.5f64.ln();
                    }
                }
            }
            _ => {
                for i in 0..n_states {
                    if i < n_states - 1 {
                        log_transition[[i, i]] = 0.5f32.ln();
                        log_transition[[i, i + 1]] = 0.5f32.ln();
                    } else {
                        log_transition[[i, i]] = 0.5f32.ln();
                    }
                }
            }
        }

        // Left-to-right topology

        let mut log_initial = vec![Float::NEG_INFINITY; n_states];
        log_initial[0] = 0.0;

        let mut log_final = vec![Float::NEG_INFINITY; n_states];
        log_final[n_states - 1] = 0.0;

        HMMGMM {
            n_states,
            state: (0..n_states).map(|_| Gmm::new(n_components, dim)).collect(),
            log_transition,
            log_initial: Array1::from_vec(log_initial),
            log_final: Array1::from_vec(log_final),

            variance_floor: Array1::zeros(dim),
        }
    }

    pub fn get_n_states(&self) -> usize {
        self.n_states
    }

    pub fn get_states(&self) -> &[Gmm] {
        &self.state
    }

    pub fn get_states_mut(&mut self) -> &mut Vec<Gmm> {
        &mut self.state
    }

    pub fn get_log_initial(&self) -> ArrayView1<'_, Float> {
        self.log_initial.view()
    }

    pub fn get_log_initial_mut(&mut self) -> ArrayViewMut1<'_, Float> {
        self.log_initial.view_mut()
    }

    pub fn get_log_transition(&self) -> ArrayView2<'_, Float> {
        self.log_transition.view()
    }

    pub fn get_log_transition_mut(&mut self) -> ArrayViewMut2<'_, Float> {
        self.log_transition.view_mut()
    }

    pub fn get_log_final(&self) -> ArrayView1<'_, Float> {
        self.log_final.view()
    }

    pub fn get_log_final_mut(&mut self) -> ArrayViewMut1<'_, Float> {
        self.log_final.view_mut()
    }

    pub fn get_variance_floor(&self) -> ArrayView1<'_, Float> {
        self.variance_floor.view()
    }

    pub fn get_variance_floor_mut(&mut self) -> ArrayViewMut1<'_, Float> {
        self.variance_floor.view_mut()
    }

    #[inline]
    /// Return log_observation vector for state s
    pub fn compute_log_emissions(
        &self,
        state: usize,
        observations: ArcArray2<Float>,
    ) -> Array1<Float> {
        cfg_select! {
            feature = "cuda" => {
                let cuda_stream = CUDA_CONTEXT.new_stream().unwrap();
                let func = CUDA_MODULE.load_function("compute_log_emissions").unwrap();

                let t_max = observations.nrows();

                let obs = observations
                    .as_standard_layout()
                    .as_slice()
                    .unwrap()
                    .to_vec();
                let mut buf_obs = cuda_stream.alloc_zeros::<Float>(obs.len()).unwrap();
                cuda_stream.memcpy_htod(&obs, &mut buf_obs).unwrap();

                let mean = self
                    .get_states()
                    .iter()
                    .flat_map(|s| {
                        (0..s.num_component())
                            .flat_map(|c| s.get_component(c).0.get_mean().to_vec())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                let mut buf_mean = cuda_stream.alloc_zeros::<Float>(mean.len()).unwrap();
                cuda_stream.memcpy_htod(&mean, &mut buf_mean).unwrap();

                let covar = self
                    .get_states()
                    .iter()
                    .flat_map(|s| {
                        (0..s.num_component())
                            .flat_map(|c| s.get_component(c).0.get_covar().to_vec())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut buf_covar = cuda_stream.alloc_zeros::<Float>(covar.len()).unwrap();
                cuda_stream.memcpy_htod(&covar, &mut buf_covar).unwrap();

                let weight = self
                    .get_states()
                    .iter()
                    .flat_map(|s| {
                        (0..s.num_component())
                            .map(|c| s.get_component(c).1)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut buf_weight = cuda_stream.alloc_zeros::<Float>(weight.len()).unwrap();
                cuda_stream.memcpy_htod(&weight, &mut buf_weight).unwrap();

                let mut buf_result_tmp = cuda_stream
                    .alloc_zeros::<Float>(t_max * self.get_states()[state].num_component())
                    .unwrap();
                let mut buf_result = cuda_stream.alloc_zeros::<Float>(t_max).unwrap();

                let cfg = LaunchConfig {
                    block_dim: (self.get_states()[0].num_component() as u32, 1, 1),
                    grid_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    cuda_stream
                        .launch_builder(&func)
                        .arg(&buf_mean)
                        .arg(&buf_covar)
                        .arg(&buf_weight)
                        .arg(&buf_obs)
                        .arg(&(self.get_states()[0].get_component(0).0.get_dimension() as u32))
                        .arg(&(self.get_states()[0].num_component() as u32))
                        .arg(&mut buf_result_tmp)
                        .arg(&mut buf_result)
                        .arg(&(state as u32))
                        .arg(&(t_max as u32))
                        .launch(cfg)
                        .unwrap()
                };
                cuda_stream.synchronize().unwrap();
                let mut result = vec![0f64; t_max];
                cuda_stream.memcpy_dtoh(&buf_result, &mut result).unwrap();

                Array1::from_vec(result)
            }
            feature = "wgpu" => {
                let (device, queue) = get_gpu_device_and_queue();

                // ── Buffers ──────────────────────────────────────────────
                let buf_obs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("obs"),
                    contents: bytemuck::cast_slice(observations.as_standard_layout().as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let buf_mean = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mean"),
                    contents: bytemuck::cast_slice(
                        &self
                            .get_states()
                            .iter()
                            .map(|s| {
                                (0..s.num_component())
                                    .map(|c| s.get_component(c).0.get_mean().to_vec())
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let buf_covar = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("covar"),
                    contents: bytemuck::cast_slice(
                        &self
                            .get_states()
                            .iter()
                            .map(|s| {
                                (0..s.num_component())
                                    .map(|c| s.get_component(c).0.get_covar().to_vec())
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let buf_weight = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("weight"),
                    contents: bytemuck::cast_slice(
                        &self
                            .get_states()
                            .iter()
                            .map(|s| {
                                (0..s.num_component())
                                    .map(|c| s.get_component(c).1)
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let result_size = (observations.nrows() * std::mem::size_of::<f32>()) as u64;

                let buf_state_id = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("state_id"),
                    contents: bytemuck::cast_slice(&[state as u32]),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let buf_result_tmp = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("result_tmp"),
                    size: (observations.nrows()
                        * std::mem::size_of::<f32>()
                        * self.get_states()[state].num_component()) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let buf_result = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("result"),
                    size: result_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: result_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut composer = Composer::default();

                composer
                    .add_composable_module(ComposableModuleDescriptor {
                        source: include_str!("wgsl/constants.wgsl"),
                        file_path: "src/wgsl/constants.wgsl",
                        ..Default::default()
                    })
                    .unwrap();

                composer
                    .add_composable_module(ComposableModuleDescriptor {
                        source: include_str!("wgsl/gmm_bindings.wgsl"),
                        file_path: "src/wgsl/gmm_bindings.wgsl",
                        ..Default::default()
                    })
                    .unwrap();

                composer
                    .add_composable_module(ComposableModuleDescriptor {
                        source: include_str!("wgsl/gmm_functions.wgsl"),
                        file_path: "src/wgsl/gmm_functions.wgsl",
                        ..Default::default()
                    })
                    .unwrap();

                let module = composer
                    .make_naga_module(NagaModuleDescriptor {
                        source: include_str!("wgsl/stt.wgsl"),
                        file_path: "src/wgsl/stt.wgsl",
                        shader_defs: HashMap::from([
                            ("GAUSSIAN_DIM".to_string(), ShaderDefValue::Int(39)),
                            (
                                "STATE_MIXTURE_COUNT".to_string(),
                                ShaderDefValue::Int(self.get_states()[0].num_component() as i32),
                            ),
                            (
                                "STATE_COUNT".to_string(),
                                ShaderDefValue::Int(self.get_n_states() as i32),
                            ),
                        ]),
                        ..Default::default()
                    })
                    .map_err(|e| {
                        println!("{}", e.emit_to_string(&composer));

                        Err::<u32, _>(e)
                    })
                    .unwrap();

                // ── Pipeline ─────────────────────────────────────────────
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("log_bj"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
                });

                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bgl"),
                    entries: &[
                        storage_entry(0, false),
                        storage_entry(1, false),
                        storage_entry(2, false),
                        storage_entry(3, true),
                        storage_entry(4, false),
                        storage_entry(5, false),
                        storage_entry(6, true),
                    ],
                });

                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("log_gauss_pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            bind_group_layouts: &[Some(&bgl)],
                            label: None,
                            ..Default::default()
                        }),
                    ),
                    module: &shader,
                    entry_point: Some("compute_log_emissions"),
                    compilation_options: PipelineCompilationOptions::default(), // {
                    cache: None,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf_mean.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buf_covar.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf_weight.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buf_obs.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: buf_result_tmp.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: buf_result.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: buf_state_id.as_entire_binding(),
                        },
                    ],
                });

                // ── Dispatch ─────────────────────────────────────────────
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut cpass = encoder.begin_compute_pass(&Default::default());
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&buf_result, 0, &buf_readback, 0, result_size);
                queue.submit(std::iter::once(encoder.finish()));

                // ── Readback ─────────────────────────────────────────────
                let slice = buf_readback.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).unwrap();
                });
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();

                let mapped = slice.get_mapped_range();
                let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
                drop(mapped);
                buf_readback.unmap();

                let output = Array1::from_vec(out);

                output
                }
            _ => {
                let (rows, _) = observations.dim();
                let mut log_emissions = vec![0.0; rows];
                log_emissions = log_emissions
                    .par_iter()
                    .with_min_len(rayon::current_num_threads())
                    .enumerate()
                    .map(|(i, _)| self.state[state].log_probability_density(observations.row(i)))
                    .collect();
                Array1::from_vec(log_emissions)
            }
        }
    }

    /// Retourne Array2<f32> n_states * obs
    pub fn get_log_observation_per_state(&self, observations: ArcArray2<Float>) -> Array2<Float> {
        cfg_select! {
            feature = "cuda" => {
                let cuda_stream = CUDA_CONTEXT.new_stream().unwrap();
                let func = CUDA_MODULE.load_function("compute_log_gmm").unwrap();

                let t_max = observations.nrows();

                let obs = observations
                    .as_standard_layout()
                    .as_slice()
                    .unwrap()
                    .to_vec();
                let mut buf_obs = cuda_stream.alloc_zeros::<Float>(obs.len()).unwrap();
                cuda_stream.memcpy_htod(&obs, &mut buf_obs).unwrap();

                let mean = self
                    .get_states()
                    .iter()
                    .flat_map(|s| {
                        (0..s.num_component())
                            .flat_map(|c| s.get_component(c).0.get_mean().to_vec())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                let mut buf_mean = cuda_stream.alloc_zeros::<Float>(mean.len()).unwrap();
                cuda_stream.memcpy_htod(&mean, &mut buf_mean).unwrap();

                let covar = self
                    .get_states()
                    .iter()
                    .flat_map(|s| {
                        (0..s.num_component())
                            .flat_map(|c| s.get_component(c).0.get_covar().to_vec())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut buf_covar = cuda_stream.alloc_zeros::<Float>(covar.len()).unwrap();
                cuda_stream.memcpy_htod(&covar, &mut buf_covar).unwrap();

                let weight = self
                    .get_states()
                    .iter()
                    .flat_map(|s| {
                        (0..s.num_component())
                            .map(|c| s.get_component(c).1)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut buf_weight = cuda_stream.alloc_zeros::<Float>(weight.len()).unwrap();
                cuda_stream.memcpy_htod(&weight, &mut buf_weight).unwrap();

                let mut buf_result_tmp = cuda_stream
                    .alloc_zeros::<Float>(t_max * self.get_n_states() * self.get_states()[0].num_component())
                    .unwrap();
                let mut buf_result = cuda_stream
                    .alloc_zeros::<Float>(t_max * self.get_n_states())
                    .unwrap();

                let cfg = LaunchConfig {
                    block_dim: (
                        self.get_states()[0].num_component() as u32,
                        self.get_n_states() as u32,
                        1,
                    ),
                    grid_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    cuda_stream
                        .launch_builder(&func)
                        .arg(&buf_mean)
                        .arg(&buf_covar)
                        .arg(&buf_weight)
                        .arg(&buf_obs)
                        .arg(&(self.get_states()[0].get_component(0).0.get_dimension() as u32))
                        .arg(&(self.get_states()[0].num_component() as u32))
                        .arg(&self.get_n_states())
                        .arg(&mut buf_result_tmp)
                        .arg(&mut buf_result)
                        .arg(&(t_max as u32))
                        .launch(cfg)
                        .unwrap()
                };
                cuda_stream.synchronize().unwrap();
                let mut result = vec![0f64; t_max * self.get_n_states()];
                cuda_stream.memcpy_dtoh(&buf_result, &mut result).unwrap();

                Array2::from_shape_vec([self.n_states, observations.nrows()], result).unwrap()
            }
            feature = "wgpu" => {
                let (device, queue) = get_gpu_device_and_queue();

                // ── Buffers ──────────────────────────────────────────────
                let buf_obs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("obs"),
                    contents: bytemuck::cast_slice(observations.as_standard_layout().as_slice().unwrap()),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let buf_mean = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mean"),
                    contents: bytemuck::cast_slice(
                        &self
                            .get_states()
                            .iter()
                            .map(|s| {
                                (0..s.num_component())
                                    .map(|c| s.get_component(c).0.get_mean().to_vec())
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let buf_covar = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("covar"),
                    contents: bytemuck::cast_slice(
                        &self
                            .get_states()
                            .iter()
                            .map(|s| {
                                (0..s.num_component())
                                    .map(|c| s.get_component(c).0.get_covar().to_vec())
                                    .flatten()
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let buf_weight = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("weight"),
                    contents: bytemuck::cast_slice(
                        &self
                            .get_states()
                            .iter()
                            .map(|s| {
                                (0..s.num_component())
                                    .map(|c| s.get_component(c).1)
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });

                let result_size =
                    (observations.nrows() * std::mem::size_of::<f32>() * self.get_n_states()) as u64;

                let buf_result_tmp = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("result_tmp"),
                    size: (observations.nrows()
                        * std::mem::size_of::<f32>()
                        * self.get_n_states()
                        * self.get_states()[0].num_component()) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let buf_result = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("result"),
                    size: result_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: result_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut composer = Composer::default();

                composer
                    .add_composable_module(ComposableModuleDescriptor {
                        source: include_str!("wgsl/constants.wgsl"),
                        file_path: "src/wgsl/constants.wgsl",
                        ..Default::default()
                    })
                    .unwrap();

                composer
                    .add_composable_module(ComposableModuleDescriptor {
                        source: include_str!("wgsl/gmm_bindings.wgsl"),
                        file_path: "src/wgsl/gmm_bindings.wgsl",
                        ..Default::default()
                    })
                    .unwrap();

                composer
                    .add_composable_module(ComposableModuleDescriptor {
                        source: include_str!("wgsl/gmm_functions.wgsl"),
                        file_path: "src/wgsl/gmm_functions.wgsl",
                        ..Default::default()
                    })
                    .unwrap();

                let module = composer
                    .make_naga_module(NagaModuleDescriptor {
                        source: include_str!("wgsl/stt.wgsl"),
                        file_path: "src/wgsl/stt.wgsl",
                        shader_defs: HashMap::from([
                            ("GAUSSIAN_DIM".to_string(), ShaderDefValue::Int(39)),
                            (
                                "STATE_MIXTURE_COUNT".to_string(),
                                ShaderDefValue::Int(self.get_states()[0].num_component() as i32),
                            ),
                            (
                                "STATE_COUNT".to_string(),
                                ShaderDefValue::Int(self.get_n_states() as i32),
                            ),
                        ]),
                        ..Default::default()
                    })
                    .map_err(|e| {
                        println!("{}", e.emit_to_string(&composer));

                        Err::<u32, _>(e)
                    })
                    .unwrap();

                // ── Pipeline ─────────────────────────────────────────────
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("log_bj"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
                });

                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bgl"),
                    entries: &[
                        storage_entry(0, false),
                        storage_entry(1, false),
                        storage_entry(2, false),
                        storage_entry(3, true),
                        storage_entry(4, false),
                        storage_entry(5, false),
                    ],
                });

                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("log_gauss_pipeline"),
                    layout: Some(
                        &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            bind_group_layouts: &[Some(&bgl)],
                            label: None,
                            ..Default::default()
                        }),
                    ),
                    module: &shader,
                    entry_point: Some("compute_log_gmm"),
                    compilation_options: PipelineCompilationOptions::default(), // {
                    //     constants: &[
                    //         (
                    //             "STATE_MIXTURE_COUNT",
                    //             self.get_states()[0].num_component() as f64,
                    //         ), // override la taille des workgroups
                    //         ("STATE_COUNT", self.get_n_states() as f64), // si ta dim est fixe (ex: MFCC = 39)
                    //     ],
                    //     zero_initialize_workgroup_memory: true,
                    //     ..Default::default()
                    // }
                    cache: None,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf_mean.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buf_covar.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf_weight.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buf_obs.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: buf_result_tmp.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: buf_result.as_entire_binding(),
                        },
                    ],
                });

                // ── Dispatch ─────────────────────────────────────────────
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut cpass = encoder.begin_compute_pass(&Default::default());
                    cpass.set_pipeline(&pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&buf_result, 0, &buf_readback, 0, result_size);
                queue.submit(std::iter::once(encoder.finish()));

                // ── Readback ─────────────────────────────────────────────
                let slice = buf_readback.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).unwrap();
                });
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();

                let mapped = slice.get_mapped_range();
                let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
                drop(mapped);
                buf_readback.unmap();

                let output = Array2::from_shape_vec([self.n_states, observations.nrows()], out).unwrap();

                output
            }
            _ => {
                let mut output = Array2::zeros([self.n_states, observations.nrows()]);
                output
                    .axis_iter_mut(Axis(0))
                    .enumerate()
                    .for_each(|(index, mut row)| {
                        row.assign(&self.compute_log_emissions(index, observations.clone()));
                    });
                output
            }
        }
    }

    /// return the next alpha of respectively each state
    pub fn compute_next_score(
        &self,
        log_alphas: ArrayView1<Float>,
        next_log_obj: ArrayView1<Float>,
    ) -> Vec<Float> {
        if log_alphas.iter().all(|&a| a == Float::NEG_INFINITY) {
            next_log_obj
                .iter()
                .enumerate()
                .map(|(state, log_obj)| self.log_initial[state] + log_obj)
                .collect()
        } else {
            next_log_obj
                .iter()
                .enumerate()
                .map(|(state, log_obj)| {
                    log_sum_exp(
                        &(0..self.n_states)
                            .map(|prev_state| {
                                log_alphas[prev_state]
                                    + self.log_transition[[prev_state, state]]
                                    + log_obj
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect()
        }
    }
}

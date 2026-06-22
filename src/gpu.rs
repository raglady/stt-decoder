cfg_select! {
    feature = "cuda" => {

        use std::sync::{Arc, LazyLock};


        use cudarc::{
            driver::{CudaContext, CudaModule},
            nvrtc::Ptx,
        };


        pub static CUDA_CONTEXT: LazyLock<Arc<CudaContext>> =
            LazyLock::new(|| CudaContext::new(0).unwrap());


        pub static KERNEL_PTX_SRC: &str = include_str!(std::env!("KERNEL_PTX_PATH"));

        pub static CUDA_MODULE: LazyLock<Arc<CudaModule>> = LazyLock::new(|| {
            CUDA_CONTEXT
                .load_module(Ptx::from_src(KERNEL_PTX_SRC))
                .unwrap()
        });

    }
    feature = "wgpu" => {
        use tokio::sync::OnceCell;
        use wgpu::{
            Adapter, Device, DeviceDescriptor, Instance, InstanceDescriptor, Queue, RequestAdapterOptions,
        };

        static GPU_ADAPTER: OnceCell<Adapter> = OnceCell::const_new();

        static GPU_DEVICE: OnceCell<Device> = OnceCell::const_new();

        static GPU_QUEUE: OnceCell<Queue> = OnceCell::const_new();

        pub async fn init_gpu_adapter() {
            if !GPU_ADAPTER.initialized() {
                let instance = Instance::new(InstanceDescriptor::new_without_display_handle());
                let adapter = instance
                    .request_adapter(&RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        ..Default::default()
                    })
                    .await
                    .unwrap();
                GPU_ADAPTER.set(adapter).unwrap();
            }
        }

        pub fn get_gpu_adapter() -> &'static Adapter {
            GPU_ADAPTER.get().unwrap()
        }

        pub async fn init_gpu_device_and_queue() {
            if !GPU_DEVICE.initialized() || !GPU_QUEUE.initialized() {
                let adapter = get_gpu_adapter();
                let limits = adapter.limits();
                let (device, queue) = adapter
                    .request_device(&DeviceDescriptor {
                        required_limits: limits,
                        ..Default::default()
                    })
                    .await
                    .unwrap();
                GPU_DEVICE.set(device).unwrap();
                GPU_QUEUE.set(queue).unwrap();
            }
        }

        pub fn get_gpu_device_and_queue() -> (&'static Device, &'static Queue) {
            (GPU_DEVICE.get().unwrap(), GPU_QUEUE.get().unwrap())
        }

        pub fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        }

        pub fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        }
    }
    _ => {

    }
}

mod data;
mod training;
use burn::{backend::{Autodiff, Wgpu}, data::{dataloader::{batcher::{self, Batcher}, Dataset}, dataset::{self, transform::Window}}, optim::AdamConfig};
use training::{train, ModelConfig, TrainingConfig};



fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let artifact_dir = "experiment";
    train::<MyAutodiffBackend>(
        "ebsentinel.db",
        &artifact_dir,
        TrainingConfig::new(ModelConfig::new(512, 128), AdamConfig::new()),
        device.clone(),
    );
}

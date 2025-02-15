mod data;
mod training;
use autoencoder::{data::{SyscallBatcher, Syscalls}, Autoencoder};
use burn::{backend::{Autodiff, Wgpu}, config::Config, data::dataloader::{batcher::Batcher, Dataset}, module::Module, nn::loss::{self, MseLoss}, optim::AdamConfig, prelude::Backend, record::{CompactRecorder, Recorder}, tensor::cast::ToElement};
use data::SyscallsDataset;
use training::{train, Model, ModelConfig, TrainingConfig};



fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let artifact_dir = "experiment";

    train::<MyAutodiffBackend>(
        "ebsentinel.db",
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(512, 64), AdamConfig::new()),
        device.clone(),
    );
    

  
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<MyBackend>(&device).load_record(record);

    let dataset=SyscallsDataset::train("ebsentinel.db");

    let mut thres: f32=0.0;

    for i in 0..dataset.len(){
        let item=dataset.get(i).unwrap();
        let (_,loss)=infer::<MyBackend>(device.clone(),&model,item.clone() );
        thres= thres.max(loss);
    }
    
    println!("threshold: {}",thres);

    println!();

}

pub fn infer<B: Backend>(device: B::Device, model: &Model<B>, item: Syscalls) -> (Vec<f32>,f32) {
    Autoencoder::infer(device,&model.inner,item)
}
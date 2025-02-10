mod data;
mod training;
use burn::{backend::{Autodiff, Wgpu}, config::Config, data::{dataloader::{batcher::{self, Batcher}, Dataset}, dataset::{self, transform::Window}}, module::Module, nn::loss::{self, MseLoss}, optim::AdamConfig, prelude::Backend, record::{CompactRecorder, Recorder}, tensor::{cast::ToElement, Tensor}};
use data::{SyscallBatcher, Syscalls, SyscallsDataset};
use training::{train, Model, ModelConfig, TrainingConfig};



fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let artifact_dir = "experiment";

    train::<MyAutodiffBackend>(
        "ebsentinel.db",
        &artifact_dir,
        TrainingConfig::new(ModelConfig::new(512, 256), AdamConfig::new()),
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
        let (out,loss)=infer::<MyBackend>(device.clone(),&model,item.clone() );
        println!("loss: {loss}");
        println!("Predicted: {:?}",out);
        println!("Expected: {:?}",item.counts);
        println!("");
        thres= thres.max(loss);
    }
    let test_item=Syscalls { counts: vec![1.0;512] };
    let (out,loss) =infer(device.clone(), &model, test_item.clone());
    println!("threshold: {}",thres);
    println!("Test");
    println!("loss: {loss}");
    println!("Predicted: {:?}",out);
    println!("Expected: {:?}",test_item);
    println!("");

}

pub fn infer<B: Backend>(device: B::Device, model: &Model<B>, item: Syscalls) -> (Vec<f32>,f32) {
    let batcher = SyscallBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.syscalls.clone());
    let loss =
        MseLoss::new().forward( output.clone(),batch.syscalls, loss::Reduction::Mean).into_scalar();

        
    (output.into_data().to_vec().unwrap(),loss.to_f32())
}
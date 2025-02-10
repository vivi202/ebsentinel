use autoencoder::{data::Syscalls, infer, AutoencoderConfig, Model};
use burn::{backend::Wgpu, config::Config, module::Module, optim::AdamConfig, prelude::Backend, record::{CompactRecorder, Recorder}};
use ebsentinel_core::{self, run_ebsentinel_ebpf};
use tokio::signal;

#[derive(Config, Debug)]
pub struct ModelConfig{
    input_size: usize, 
    latent_size: usize
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let inner=AutoencoderConfig::new(self.input_size, self.latent_size).init(device);
        Model { inner }
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 200)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

//Main program uses the previusly trained model to detect anomalies
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    type MyBackend = Wgpu<f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let mut proc_mon=run_ebsentinel_ebpf(163389).unwrap();
    let mut rx =proc_mon.run().unwrap();
    
    let artifact_dir = "experiment";
    //TODO Load only model since TrainingConfig is useless
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
    .expect("Config should exist for the model");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");


    let model = config.model.init::<MyBackend>(&device).load_record(record);
    
    tokio::spawn(async move {
        
        loop {
            let rates=rx.recv().await.unwrap();
            let item= Syscalls { counts: rates };
            //Infer
            let (out, loss) = infer(device.clone(), &model, item);
            println!("{}",loss);
            
            if loss > 0.000015 {
                println!("anomaly detected")
            }
        }
    });
    let ctrl_c = signal::ctrl_c();
    println!("Waiting for Ctrl-C...");
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
}


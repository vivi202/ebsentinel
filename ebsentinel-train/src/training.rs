use std::vec;

use autoencoder::{Autoencoder, AutoencoderConfig};
use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::LossMetric,
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

use crate::data::{SyscallBatch, SyscallBatcher, SyscallsDataset};

use burn::tensor::{backend::Backend, Tensor};

#[derive(Module,Debug)]
pub struct Model<B: Backend>{
    inner: Autoencoder<B>
}

impl<B: Backend> Model<B> {
    pub fn forward_reconstruction(&self, vecs: Tensor<B, 2>) -> RegressionOutput<B> {
        let output = self.inner.forward(vecs.clone());
        let loss: Tensor<B, 1> =
            MseLoss::new().forward( output.clone(),vecs.clone(), nn::loss::Reduction::Mean);
        RegressionOutput::new(loss, output, vecs)
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, vecs: Tensor<B, 2>) -> Tensor<B,2> {
       self.inner.forward(vecs)
    }
}

impl<B: AutodiffBackend> TrainStep<SyscallBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: SyscallBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_reconstruction(batch.syscalls);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}


impl<B: Backend> ValidStep<SyscallBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: SyscallBatch<B>) -> RegressionOutput<B> {
        self.forward_reconstruction(batch.syscalls)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig{
    input_size: usize, 
    latent_size: usize
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let inner=AutoencoderConfig::new(self.input_size,self.latent_size).init(device);
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
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    db_file: &str,
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = SyscallBatcher::<B>::new(device.clone());
    let batcher_valid = SyscallBatcher::<B::InnerBackend>::new(device.clone());

    let train_dataset = SyscallsDataset::train(db_file);
    let test_dataset = SyscallsDataset::test(db_file);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained:Model<B> = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

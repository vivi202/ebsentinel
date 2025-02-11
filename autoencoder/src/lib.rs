use burn::nn::loss::{self, MseLoss};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::cast::ToElement;
use burn::data::dataloader::batcher::Batcher;
use data::{SyscallBatcher, Syscalls};
pub mod data;

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    linear: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
    activation2: Relu,
}

impl<B: Backend> Encoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(input);
        let x=self.activation.forward(x);
        let x=self.linear2.forward(x);
        self.activation2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Middle<B: Backend> {
    linear: Linear<B>,
    norm: LayerNorm<B>,
    activation: Relu,
}

impl<B: Backend> Middle<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x=self.linear.forward(input);
        let x= self.norm.forward(x);
        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    linear: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
    activation2: Relu,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(input);
        let x = self.activation.forward(x);
        let x: Tensor<B, 2> = self.linear2.forward(x);
        self.activation2.forward(x)

    }
}

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: Encoder<B>,
    middle: Middle<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> Autoencoder<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let encoded = self.encoder.forward(input);
        let transformed = self.middle.forward(encoded);
        self.decoder.forward(transformed)
    }
}

#[derive(Config, Debug)]
pub struct AutoencoderConfig {
    input_size: usize,
    latent_size: usize,
}

impl AutoencoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Autoencoder<B> {
        Autoencoder {
            encoder: Encoder {
                linear: LinearConfig::new(self.input_size, self.input_size/2).init(device),
                activation: Relu::new(),
                linear2: LinearConfig::new(self.input_size/2, self.latent_size).init(device),
                activation2: Relu::new(),

            },
            middle: Middle {
                linear: LinearConfig::new(self.latent_size, self.latent_size).init(device),
                //This help to stabilize training.
                norm: LayerNormConfig::new(self.latent_size).init(device),
                activation: Relu::new(),
            },
            decoder: Decoder {
                linear: LinearConfig::new(self.latent_size, self.input_size/2).init(device),
                activation: Relu::new(),
                linear2: LinearConfig::new(self.input_size/2, self.input_size).init(device),
                activation2: Relu::new(),
            },
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub inner: Autoencoder<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    input_size: usize,
    latent_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let inner = AutoencoderConfig::new(self.input_size, self.latent_size).init(device);
        Model { inner }
    }
}

pub fn infer<B: Backend>(device: B::Device, model: &Model<B>, item: Syscalls) -> (Vec<f32>, f32) {
    let batcher = SyscallBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.inner.forward(batch.syscalls.clone());
    let loss = MseLoss::new()
        .forward(batch.syscalls, output.clone(), loss::Reduction::Mean)
        .detach()  
        .into_scalar();

    (output.into_data().to_vec().unwrap(), loss.to_f32())
}

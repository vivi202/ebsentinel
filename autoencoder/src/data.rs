use burn::{data::dataloader::batcher::Batcher, prelude::*};
use serde::{Deserialize, Serialize};
#[derive(Clone)]
pub struct SyscallBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SyscallBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}
#[derive(Clone, Debug)]

pub struct SyscallBatch<B: Backend> {
    //[batch_size,MAX_SYSCALLS]
    pub syscalls: Tensor<B, 2>,
}


impl<B: Backend> Batcher<Syscalls, SyscallBatch<B>> for SyscallBatcher<B> {
    fn batch(&self, items: Vec<Syscalls>) -> SyscallBatch<B> {
        let vecs = items
            .iter()
            .map(|item| {
                // Convert vector into TensorData, then create a Tensor with shape [1, 5]
                let data = TensorData::from(item.counts.as_slice()).convert::<B::FloatElem>();
                let tensor = Tensor::<B, 1>::from_data(data, &self.device);
                tensor.reshape([1, item.counts.len()]) // Reshape each tensor to [1, 5]
            })
            .collect::<Vec<_>>();
        let syscalls_batch = Tensor::cat(vecs, 0).to_device(&self.device);
        SyscallBatch { syscalls: syscalls_batch }
    }
}



#[derive(Debug, Clone,Serialize,Deserialize)]
struct SyscallsRaw{
    #[serde(with="serde_bytes")]
    syscalls: Vec<u8>
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Syscalls{
    pub counts: Vec<f32>
}





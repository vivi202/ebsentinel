use std::time::Duration;

use anyhow::Ok;
use ebsentinel_common::MAX_SYSCALLS;

pub trait DataProcessor<IN,OUT>{
    fn process(&mut self,data: IN ) -> anyhow::Result<OUT>;
}

pub struct Differentiator {
    frequency: f32,
    prev: Vec<u64>
}

impl Differentiator{
    pub fn new(polling_rate: &Duration) -> Self{
        Self{
            frequency: 1.0/polling_rate.as_secs_f32(),
            prev: vec![0; MAX_SYSCALLS as usize]
        }
    
    }
}

impl DataProcessor<&[u64],Vec<f32>> for Differentiator {
    fn process(&mut self,data: &[u64] ) -> anyhow::Result<Vec<f32>> {
        let rates : Vec<f32>= data.iter().enumerate().map(|(idx,value)| (*value as f32 - self.prev[idx] as f32) * self.frequency ).collect();
        self.prev=data.to_vec();
        Ok(rates)
    }
}
pub struct Normalizer;

impl DataProcessor<&[f32],Vec<f32>> for Normalizer{
    fn process(&mut self,data: &[f32] ) -> anyhow::Result<Vec<f32>> {
        let max=data.iter().fold(0.0f32, |a, &b| a.max(b));
        if max == 0.0 {
            return Ok(data.to_vec());
        } 
        let norm : Vec<f32>=data.iter().map(|x| x/max).collect();
        Ok(norm)

    }
}
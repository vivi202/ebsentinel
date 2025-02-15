use autoencoder::data::Syscalls;
use burn::data::dataset::{transform::{Mapper, MapperDataset}, Dataset, SqliteDataset};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone,Serialize,Deserialize)]
struct SyscallsRaw{
    #[serde(with="serde_bytes")]
    syscalls: Vec<u8>
}

struct SyscallsRawToSyscalls;

impl Mapper<SyscallsRaw, Syscalls> for SyscallsRawToSyscalls {
    /// Convert a raw syscall to Syscall
    fn map(&self, item: &SyscallsRaw) -> Syscalls {
        // Ensure the image dimensions are correct.

        Syscalls {
            counts: bincode::deserialize(&item.syscalls).unwrap()
        }
    }
}

type MappedDataset = MapperDataset<SqliteDataset<SyscallsRaw>,SyscallsRawToSyscalls,SyscallsRaw>;

pub struct SyscallsDataset{
    dataset: MappedDataset
}

impl SyscallsDataset {
    /// Creates a new train dataset.
    pub fn train(db_file: &str) -> Self {
        Self::new(db_file,"train")
    }

    /// Creates a new test dataset.
    pub fn test(db_file: &str) -> Self {
        Self::new(db_file,"test")
    }

    fn new(db_file: &str,split: &str) -> Self {
        let dataset_raw: SqliteDataset<SyscallsRaw> = SqliteDataset::from_db_file(db_file, split).unwrap();

        // Create the MapperDataset for InMemDataset<MnistItemRaw> to transform
        // items (MnistItemRaw -> MnistItem)
        let dataset = MapperDataset::new(dataset_raw, SyscallsRawToSyscalls);

        Self { dataset }
    }

}

impl Dataset<Syscalls> for SyscallsDataset{
    fn get(&self, index: usize) -> Option<Syscalls> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[cfg(test)]
mod test{
    use std::num::NonZero;

    use burn::{backend::Wgpu, data::{dataloader::batcher::Batcher, dataset::{transform::Window, Dataset}}};

    use autoencoder::data::SyscallBatcher;

    use super::SyscallsDataset;
    #[test]
    pub fn it_works(){
        type MyBackend = Wgpu<f32, i32>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let dataset= SyscallsDataset::train("ebsentinel.db");
        let items = dataset.window(0, NonZero::new(dataset.len()).unwrap()).unwrap();
        let batcher: SyscallBatcher<MyBackend> = SyscallBatcher::new(device);
        let batch = batcher.batch(items.clone());
        let syscals :Vec<Vec<f32>>=batch.syscalls.iter_dim(0).map(|x| x.to_data().to_vec().unwrap()).collect();
        let counts: Vec<Vec<f32>>=items.iter().map(|i|i.counts.clone()).collect();
        assert_eq!(counts,syscals)
    }
}
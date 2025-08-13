
// loader.rs
use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Tensor},
};

#[derive(Clone , Debug)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MnistItem {
    pub image: Vec<f32>,
    pub label: usize,
}

#[derive(Debug, Clone)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, burn::tensor::Int>,
}

// Fix: Correct Batcher implementation with 3 parameters
impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items.iter().flat_map(|item| item.image.clone()).collect::<Vec<_>>();
        let labels = items.iter().map(|item| item.label as i64).collect::<Vec<_>>();

        MnistBatch {
            // Fix: Convert Vec to slice reference for TensorData
            images: Tensor::from_floats(images.as_slice(), device)
                .reshape([items.len(), 28, 28]),
            targets: Tensor::from_ints(labels.as_slice(), device),
        }
    }
}
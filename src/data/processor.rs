// processor.rs
use burn::{
    data::{dataset::Dataset, dataloader::{DataLoader, DataLoaderBuilder}},
    tensor::{backend::Backend, Tensor, Data},
};
use std::path::Path;
use super::loader::{MnistBatcher, MnistItem, MnistBatch};

/// Trait pour le prétraitement des données
pub trait DataProcessor<B: Backend> {
    fn process_images(&self, images: Vec<Vec<f32>>) -> Tensor<B, 3>;
    fn process_labels(&self, labels: Vec<usize>) -> Tensor<B, 1, burn::tensor::Int>;
    fn normalize(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3>;
}

/// Implémentation concrète du processeur MNIST
#[derive(Debug, Clone)]
pub struct MnistProcessor<B: Backend> {
    device: B::Device,
    normalize_mean: f32,
    normalize_std: f32,
}

impl<B: Backend> MnistProcessor<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            normalize_mean: 0.1307,
            normalize_std: 0.3081,
        }
    }
}

impl<B: Backend> DataProcessor<B> for MnistProcessor<B> {
    fn process_images(&self, images: Vec<Vec<f32>>) -> Tensor<B, 3> {
        let flat_images: Vec<f32> = images.clone().into_iter().flatten().collect();
        let batch_size = images.len();
        
        // Fix: Convert Vec to slice reference for TensorData
        Tensor::from_floats(flat_images.as_slice(), &self.device)
            .reshape([batch_size, 28, 28])
    }

    fn process_labels(&self, labels: Vec<usize>) -> Tensor<B, 1, burn::tensor::Int> {
        // Fix: Convert Vec to slice reference for TensorData
        let labels_i64: Vec<i64> = labels.into_iter().map(|l| l as i64).collect();
        Tensor::from_ints(labels_i64.as_slice(), &self.device)
    }

    fn normalize(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        (tensor - self.normalize_mean) / self.normalize_std
    }
}

/// Charge et prétraite le dataset MNIST
pub fn process_mnist<B: Backend>(
    batch_size: usize,
) -> (
    // Fix: Add missing generic parameter for DataLoader
    DataLoaderBuilder<MnistBatcher<B>, MnistBatch<B>>,
    DataLoaderBuilder<MnistBatcher<B>, MnistBatch<B>>,
) {
    let device = B::Device::default();
    let batcher_train = MnistBatcher::new(device.clone());
    let batcher_valid = MnistBatcher::new(device);

    let train_data = load_mnist_data("data/mnist/train-images-idx3-ubyte");
    let train_labels = load_mnist_labels("data/mnist/train-labels-idx1-ubyte");
    let test_data = load_mnist_data("data/mnist/t10k-images-idx3-ubyte");
    let test_labels = load_mnist_labels("data/mnist/t10k-labels-idx1-ubyte");

    let train_dataset = MnistDataset::new(train_data, train_labels);
    let valid_dataset = MnistDataset::new(test_data, test_labels);

    (
        // Fix: Use concrete types instead of trait objects
        DataLoaderBuilder::new(batcher_train)
            .batch_size(batch_size)
            .build(train_dataset),
        DataLoaderBuilder::new(batcher_valid)
            .batch_size(batch_size)
            .build(valid_dataset),
    )
}

fn load_mnist_data<P: AsRef<Path>>(_path: P) -> Vec<Vec<f32>> {
    vec![vec![0.0; 784]; 60000] // Placeholder
}

fn load_mnist_labels<P: AsRef<Path>>(_path: P) -> Vec<usize> {
    vec![0; 60000] // Placeholder
}

#[derive(Debug, Clone)]
pub struct MnistDataset {
    items: Vec<MnistItem>,
}

impl MnistDataset {
    pub fn new(images: Vec<Vec<f32>>, labels: Vec<usize>) -> Self {
        let items = images
            .into_iter()
            .zip(labels)
            .map(|(image, label)| MnistItem { image, label })
            .collect();
        Self { items }
    }
}

impl Dataset<MnistItem> for MnistDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

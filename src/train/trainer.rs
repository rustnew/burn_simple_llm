use burn::{
    config::Config,
    module::Module,
    optim::AdamConfig,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::backend::{ADBackend, Backend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

use crate::{
    data::{loader::MnistBatcher, processor::process_mnist},
    model::{config::ModelConfig, network::{MLP, MLPConfig}},
};

pub fn train<B: ADBackend>(device: B::Device, config: ModelConfig) {
    // Créer le modèle
    let model = MLPConfig::new(
        config.input_size,
        config.hidden_size,
        config.output_size,
        config.dropout_rate,
    )
    .init::<B>();

    // Optimiseur
    let optim = AdamConfig::new().init();

    // Charger les données
    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B>::new(device.clone());
    
    let (train_dataset, valid_dataset) = process_mnist(config.batch_size);

    // Configurer l'entraînement
    let learner = LearnerBuilder::new(&device)
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer::<FullPrecisionSettings>(2, "model")
        .num_epochs(config.epochs)
        .build(model, optim);

    // Lancer l'entraînement
    let trained_model = learner.fit(
        train_dataset,
        valid_dataset,
        batcher_train,
        batcher_valid,
    );

    // Sauvegarder le modèle final
    BinFileRecorder::<FullPrecisionSettings>::new()
        .record(
            trained_model.into_record(),
            "trained_model".into(),
        )
        .unwrap();
}
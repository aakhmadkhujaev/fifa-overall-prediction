import pytest
import torch
import numpy as np
import pandas as pd
from src.training.dataset import FIFAPlayerDataset
from src.training.dataloader import create_dataloaders
from src.preprocessing.preprocessor import Preprocessor

@pytest.fixture
def synthetic_numerical_data():
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)

    return X, y

def test_dataset_length_and_structure(synthetic_numerical_data):
    X, y = synthetic_numerical_data
    dataset = FIFAPlayerDataset(X, y)

    # 1. Dataset length
    assert len(dataset) == 100

    # 2. Dataset item structure
    features, target = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(target, torch.Tensor)

def test_dataset_tensor_types_and_shapes(synthetic_numerical_data):
    X, y = synthetic_numerical_data
    dataset = FIFAPlayerDataset(X, y)
    features, target = dataset[0]

    # 3. Feature tensor dtype
    assert features.dtype == torch.float32

    # 4. Target tensor dtype
    assert target.dtype == torch.float32

    # 5. Feature tensor shape
    assert features.shape == (10,)

    # 6. Target tensor shape
    assert target.shape == (1,)

def test_mismatched_lengths():
    X = np.random.randn(100, 10)
    y = np.random.randn(99)

    # 10. Mismatched feature/target lengths
    with pytest.raises(ValueError, match="does not match target rows"):
        FIFAPlayerDataset(X, y)

def test_dataloader_configuration(synthetic_numerical_data):
    X, y = synthetic_numerical_data
    X_dict = {'train': X, 'val': X[:20], 'test': X[20:40]}
    y_dict = {'train': y, 'val': y[:20], 'test': y[20:40]}

    batch_size = 16
    dataloaders = create_dataloaders(X_dict, y_dict, batch_size=batch_size)

    assert 'train' in dataloaders
    assert 'val' in dataloaders
    assert 'test' in dataloaders

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # 8. Training DataLoader shuffling configuration
    # DataLoader internally stores this in sampler
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)

    # 9. Validation/test DataLoader behavior
    assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler)
    assert isinstance(dataloaders['test'].sampler, torch.utils.data.SequentialSampler)

    # 7. DataLoader batch dimensions
    train_iter = iter(train_loader)
    batch_X, batch_y = next(train_iter)

    assert batch_X.shape == (16, 10)
    assert batch_y.shape == (16, 1)

def test_integration_preprocessing_to_dataloader():
    # 11. Integration test demonstrating:
    # Preprocessed numerical data -> PyTorch Dataset -> DataLoader -> batch
    np.random.seed(42)
    n_samples = 50
    df = pd.DataFrame({
        'player_id': np.arange(n_samples),
        'overall': np.random.randint(50, 90, n_samples),
        'preferred_foot': np.random.choice(['Right', 'Left'], n_samples),
        'power_stamina': np.random.rand(n_samples) * 100,
        'skill_dribbling': np.random.rand(n_samples) * 100
    })

    preprocessor = Preprocessor()
    X_dict, y_dict, stats = preprocessor.process(df)

    # Preprocessed dicts go to dataloaders
    batch_size = 4
    dataloaders = create_dataloaders(X_dict, y_dict, batch_size=batch_size)

    train_loader = dataloaders['train']
    batch_X, batch_y = next(iter(train_loader))

    # Verify expected shape and dtype
    assert batch_X.shape[0] == batch_size
    # Features count: preferred_foot, power_stamina, skill_dribbling = 3 features
    assert batch_X.shape[1] == 3
    assert batch_y.shape == (batch_size, 1)

    assert batch_X.dtype == torch.float32
    assert batch_y.dtype == torch.float32

# test_calculator.py

import pytest
from synapselab.engine.config import Configuration


# A fixture providing a common set of numbers for multiple tests.
@pytest.fixture
def sample_config() -> Configuration:
    config_path = "tests/sample_config.yaml"
    configuration = Configuration(config_path)
    return configuration

def test_get_dataset(sample_config):
    assert sample_config.dataset == "JSC"
    
def test_get_epochs(sample_config):
    assert sample_config.epochs == 70
    
def test_set_attr_backbone(sample_config):
    sample_config.set_layer_attribute("backbone", "dense1", "in_features", 3)
    assert sample_config.backbone[0][4]["in_features"] == 3
    
def test_set_attr_backbone_error(sample_config):
    with pytest.raises(ValueError):
        sample_config.set_layer_attribute("backbone", "dense2", "features_in", 3)

def test_set_attr_backbone_force(sample_config):
    sample_config.set_layer_attribute("backbone", "dense2", "features_in", 3, force=True)
    print(sample_config.backbone[2][4])
    assert sample_config.backbone[2][4]["features_in"] == 3

def test_set_attr_backbone_multiple_layers(sample_config):
    sample_config.set_layer_attribute("backbone", "dense", "in_features", 3)
    assert sample_config.backbone[0][4]["in_features"] == 3
    assert sample_config.backbone[2][4]["in_features"] == 3
    
def test_set_remove_attr_backbone(sample_config):
    # Set attribute
    sample_config.set_layer_attribute("backbone", "dense1", "test_attribute", 3, force=True)
    assert sample_config.backbone[0][4]["test_attribute"] == 3
    
    # Remove attribute
    sample_config.remove_layer_attribute("backbone", "dense1", "test_attribute")
    assert "test_attribute" not in sample_config.backbone[0][4]
    
def test_export_config(sample_config):
    sample_config.export("tests/sample_config_export.yaml")
    exported_config = Configuration("tests/sample_config_export.yaml")
    assert sample_config.__dict__ == exported_config.__dict__

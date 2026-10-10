import pytest
import numpy as np
import pandas as pd
from src.architecture_synthesizer import ArchitectureSynthesizer

def test_fold_architecture_structure(sample_requirements_df):
    synthesizer = ArchitectureSynthesizer(target_modules=2)
    J = np.zeros((7, 7))
    sigma = np.ones(7)
    blueprint = synthesizer.fold_architecture(sample_requirements_df, J, sigma)

    assert blueprint['total_requirements'] == 7
    assert blueprint['functional_count'] == 2
    assert blueprint['nfr_count'] == 5
    assert len(blueprint['synthesized_modules']) <= 2
    assert 'metrics' in blueprint
    metrics = blueprint['metrics']
    assert 0.0 <= metrics['cohesion_modularity'] <= 1.0
    assert 0.0 <= metrics['coupling_density'] <= 1.0
    assert 0.0 <= metrics['constraint_satisfaction_rate'] <= 1.0
    assert 0.0 <= metrics['architecture_fitness'] <= 1.0

def test_infrastructure_enclave_assignment(sample_requirements_df):
    synthesizer = ArchitectureSynthesizer()
    nfr_df = sample_requirements_df[sample_requirements_df['Type'] != 'F']
    sigma = np.ones(len(nfr_df))
    enclaves = synthesizer._synthesize_infrastructure(nfr_df, sigma)

    # SE and US should map to Edge_API_Gateway_Auth
    assert 'Edge_API_Gateway_Auth' in enclaves
    # PE and SC should map to In_Memory_Cache_Redis
    assert 'In_Memory_Cache_Redis' in enclaves
    # A should map to High_Availability_LoadBalancer
    assert 'High_Availability_LoadBalancer' in enclaves

def test_empty_functional_handling():
    synthesizer = ArchitectureSynthesizer()
    empty_f_df = pd.DataFrame(columns=['Clean_Requirement'])
    modules = synthesizer._cluster_functional_modules(empty_f_df)
    assert 'Core_General_Service' in modules

def test_export_docker_compose(sample_requirements_df, tmp_path):
    synthesizer = ArchitectureSynthesizer(target_modules=2)
    J = np.zeros((7, 7))
    sigma = np.ones(7)
    blueprint = synthesizer.fold_architecture(sample_requirements_df, J, sigma)
    out_file = tmp_path / "docker-compose.yml"
    synthesizer.export_docker_compose(blueprint, str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert "version: '3.8'" in content
    assert "services:" in content
    assert "redis-cache:" in content
    assert "edge-api-gateway:" in content

def test_export_docker_compose_without_gateway(tmp_path):
    synthesizer = ArchitectureSynthesizer()
    blueprint = {
        'infrastructure_enclaves': {'In_Memory_Cache_Redis': [{'type': 'PE', 'text': 'Fast caching'}]},
        'synthesized_modules': {'Worker_Service': [{'req_idx': 0, 'text': 'Process tasks'}]}
    }
    out_file = tmp_path / "docker-compose-no-gateway.yml"
    synthesizer.export_docker_compose(blueprint, str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert "worker-service:" in content
    assert "depends_on:" not in content
    assert "redis-cache:" in content

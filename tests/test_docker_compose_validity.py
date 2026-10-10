import yaml
import sys
import os

def test_docker_compose_syntax_and_schema():
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker", "docker-compose.yml")
    assert os.path.exists(yaml_path), f"File not found: {yaml_path}"
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    # 1. Structural schema assertions
    assert "version" in compose_data or "services" in compose_data
    assert "services" in compose_data
    assert "networks" in compose_data
    
    services = compose_data["services"]
    assert len(services) >= 3, "Should define gateway, cache, and microservices"
    
    # 2. Key enclave components exist
    assert "edge-api-gateway" in services
    assert "redis-cache" in services
    
    # 3. Port & network connectivity validation
    assert "ports" in services["edge-api-gateway"]
    assert "ports" in services["redis-cache"]
    assert "app-network" in compose_data["networks"]
    
    for s_name, s_def in services.items():
        assert "image" in s_def, f"Service {s_name} missing image tag"
        assert "networks" in s_def, f"Service {s_name} missing network allocation"
        assert "app-network" in s_def["networks"]

def test_generated_project_8_compose_file():
    yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "project_8_docker_compose.yml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            compose_data = yaml.safe_load(f)
        assert "services" in compose_data
        assert "edge-api-gateway" in compose_data["services"]

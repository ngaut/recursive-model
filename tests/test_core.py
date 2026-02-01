
import pytest
import torch
from rva.config import RVAConfig
from rva.genome_kernel import GenomeKernel, VariantModulation
from rva.variant_generator import VariantGenerator
from rva.recursive_engine import RecursiveEngine
from rva.self_improve import SelfImprovementEngine

@pytest.fixture
def config():
    return RVAConfig(
        state_dim=32,
        hidden_dim=64,
        memory_dim=16,
        variant_code_dim=8,
        num_variant_prototypes=4,
        max_recursion_depth=10,
        min_recursion_depth=2,
        input_dim=10,
        output_dim=10
    )

def test_genome_kernel_shape(config):
    kernel = GenomeKernel(config)
    batch_size = 5
    state = torch.randn(batch_size, config.state_dim)
    memory = torch.randn(batch_size, config.memory_dim)
    
    # Mock modulation
    gammas = [torch.ones(batch_size, config.hidden_dim) for _ in range(config.num_kernel_layers)]
    betas = [torch.zeros(batch_size, config.hidden_dim) for _ in range(config.num_kernel_layers)]
    context = torch.randn(batch_size, config.variant_code_dim)
    modulation = VariantModulation(gammas, betas, context)
    
    output = kernel(state, memory, modulation)
    
    assert output.state.shape == (batch_size, config.state_dim)
    assert output.memory.shape == (batch_size, config.memory_dim)
    assert output.halt_logit.shape == (batch_size, 1)
    assert output.improvement.shape == (batch_size, config.variant_code_dim)

def test_variant_generator_shape(config):
    vg = VariantGenerator(config)
    batch_size = 5
    state = torch.randn(batch_size, config.state_dim)
    depth = torch.randint(0, 10, (batch_size,))
    
    modulation = vg(state, depth)
    
    assert len(modulation.gammas) == config.num_kernel_layers
    assert len(modulation.betas) == config.num_kernel_layers
    assert modulation.context.shape == (batch_size, config.variant_code_dim)

def test_recursive_engine_flow(config):
    engine = RecursiveEngine(config)
    batch_size = 3
    state = torch.randn(batch_size, config.state_dim)
    memory = torch.randn(batch_size, config.memory_dim)
    
    output = engine(state, memory)
    
    assert output.final_state.shape == (batch_size, config.state_dim)
    assert output.num_steps.shape == (batch_size,)
    assert (output.num_steps >= config.min_recursion_depth).all()
    assert (output.num_steps <= config.max_recursion_depth).all()

def test_self_improvement_update(config):
    sie = SelfImprovementEngine(config)
    batch_size = 5
    # Signal is now [B, K, D] (distributed per prototype)
    signal = torch.randn(batch_size, config.num_variant_prototypes, config.variant_code_dim)
    
    # Check update computation
    updates, plasticity = sie.compute_update(signal)
    assert updates.shape == (config.num_variant_prototypes, config.variant_code_dim)
    assert plasticity.shape == (config.num_variant_prototypes,)

def test_grouped_consistency_loss(config):
    sie = SelfImprovementEngine(config)
    batch_size = 4
    signal = torch.randn(batch_size, config.variant_code_dim)
    
    # Case 1: No groups (default behavior)
    loss_all = sie.get_improvement_loss(signal)
    
    # Case 2: With groups
    group_ids = torch.tensor([0, 0, 1, 1])
    loss_grouped = sie.get_improvement_loss(signal, group_ids=group_ids)
    
    assert loss_all.item() != 0
    assert loss_grouped.item() != 0

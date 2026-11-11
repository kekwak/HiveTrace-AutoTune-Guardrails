"""Standalone U1 measurements. Importing this package does not load CUDA."""

def run(config, output_dir):
    from .orchestrator import run as execute
    return execute(config, output_dir)

import os
import torch
import numpy as np
from lidarsemseg.engines.test import TESTERS
from lidarsemseg.engines.defaults import default_argument_parser, default_config_parser

def main():
    # Parse arguments and config
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    
    # Set batch size to 1 for testing
    cfg.batch_size_test_per_gpu = 1
    
    # Ensure test configuration exists
    if not hasattr(cfg, 'test'):
        cfg.test = type('TestConfig', (), {})()
    cfg.test.type = "SemSegTester"
    
    # Set weight path from command line argument
    if args.weight:
        cfg.weight = args.weight
    elif not hasattr(cfg, 'weight'):
        raise ValueError("No weight path specified. Use --weight argument or set in config")
    
    # Ensure weight path exists
    if not os.path.exists(cfg.weight):
        raise FileNotFoundError(f"Weight file not found at: {cfg.weight}")
    
    # Create save directory if it doesn't exist
    os.makedirs(cfg.save_path, exist_ok=True)
    
    # Initialize tester
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    
    # Run test
    tester.test()

if __name__ == "__main__":
    main()
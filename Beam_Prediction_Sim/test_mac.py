#!/usr/bin/env python3
"""
맥북 호환성 테스트 스크립트
Mac Compatibility Test Script
"""
import sys
import torch

def test_imports():
    """Test if all imports work"""
    print("="*70)
    print("1. Testing imports...")
    print("="*70)
    
    try:
        from config import Config
        print("✓ config module imported")
        
        from utils import set_seed
        print("✓ utils module imported")
        
        from data import BeamSeqDataset
        print("✓ data module imported")
        
        from models import BeamPredictorLLM
        print("✓ models module imported")
        
        print("\n✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n❌ Import error: {e}\n")
        return False


def test_device():
    """Test device availability"""
    print("="*70)
    print("2. Testing device availability...")
    print("="*70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("  ⚠️  MPS is available but experimental")
            print("  ✅ Recommended: use --device cpu for stability")
    else:
        print("MPS: Not available (PyTorch < 2.0)")
    
    print(f"\n✅ Device test complete!\n")


def test_config():
    """Test configuration"""
    print("="*70)
    print("3. Testing configuration...")
    print("="*70)
    
    try:
        from config import Config, get_lightweight_config
        
        cfg = get_lightweight_config()
        print(f"Lightweight config created:")
        print(f"  - num_workers: {cfg.num_workers}")
        print(f"  - pin_memory: {cfg.pin_memory}")
        print(f"  - use_revin: {cfg.use_revin}")
        print(f"  - batch_size: {cfg.batch_size}")
        print(f"  - device: {cfg.device}")
        
        device = cfg.get_device()
        print(f"\nSelected device: {device}")
        
        print("\n✅ Configuration test complete!\n")
        return True
    except Exception as e:
        print(f"\n❌ Config error: {e}\n")
        return False


def test_model():
    """Test model creation"""
    print("="*70)
    print("4. Testing model creation...")
    print("="*70)
    
    try:
        from config import get_lightweight_config
        from models import BeamPredictorLLM
        
        cfg = get_lightweight_config()
        cfg.use_gpt2 = False  # Disable GPT-2 for quick test
        
        print("Creating model (without GPT-2)...")
        model = BeamPredictorLLM(cfg)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        print("\n✅ Model test complete!\n")
        return True
    except Exception as e:
        print(f"\n❌ Model error: {e}\n")
        return False


def test_tensor_ops():
    """Test tensor operations on different devices"""
    print("="*70)
    print("5. Testing tensor operations...")
    print("="*70)
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    
    for device_str in devices:
        try:
            device = torch.device(device_str)
            print(f"\nTesting on {device_str}...")
            
            # Simple tensor operations
            x = torch.randn(2, 10, 40).to(device)
            x = x.contiguous()  # MPS compatibility
            
            # Conv1D
            conv = torch.nn.Conv1d(10, 32, kernel_size=5, stride=2).to(device)
            y = conv(x)
            y = y.contiguous()
            
            # Transpose
            y = y.transpose(1, 2)
            y = y.contiguous()
            
            print(f"  ✓ {device_str}: Basic operations successful")
            print(f"    Input shape: {x.shape}")
            print(f"    Output shape: {y.shape}")
            
        except Exception as e:
            print(f"  ⚠️  {device_str}: {e}")
            if device_str == 'mps':
                print(f"      Recommendation: use --device cpu")
    
    print("\n✅ Tensor operations test complete!\n")


def print_recommendations():
    """Print final recommendations"""
    print("="*70)
    print("RECOMMENDATIONS FOR MAC USERS")
    print("="*70)
    
    print("\n📝 Quick Start Commands:\n")
    
    print("1. Lightweight test (5-10 min, SAFE):")
    print("   python main.py --config lightweight --epochs 5 --mode both --device cpu\n")
    
    print("2. With smaller batch size (even safer):")
    print("   python main.py --config lightweight --epochs 5 --mode both --device cpu --batch_size 8\n")
    
    print("3. Try MPS (experimental):")
    print("   python main.py --config lightweight --epochs 5 --mode both --device mps --batch_size 8\n")
    
    print("📚 Documentation:")
    print("   - Full guide: README.md")
    print("   - Quick start: QUICKSTART.md")
    print("   - Mac guide: MPS_GUIDE.md ⭐")
    print("   - Fix details: MPS_FIX.md ⭐\n")
    
    print("✅ All tests completed!")
    print("="*70)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("MAC COMPATIBILITY TEST")
    print("="*70 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    test_device()
    results.append(("Config", test_config()))
    results.append(("Model", test_model()))
    test_tensor_ops()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All critical tests passed!")
        print_recommendations()
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("For help, see MPS_GUIDE.md\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

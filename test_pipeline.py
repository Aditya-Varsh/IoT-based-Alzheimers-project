"""
Simple test to verify the pipeline works
"""

def test_import():
    try:
        from main import SisFallPredictor
        print("✓ Successfully imported SisFallPredictor from main.py")
        return True
    except ImportError as e:
        print(f"✗ Failed to import SisFallPredictor: {e}")
        return False

def test_packages():
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        print("✓ Required packages (numpy, pandas, sklearn) are available")
        return True
    except ImportError as e:
        print(f"✗ Missing required packages: {e}")
        return False

def test_dataset():
    import os
    data_path = r"Sisfall Dataset\SisFall_dataset"
    if os.path.exists(data_path):
        print("✓ Dataset directory found")
        return True
    else:
        print("✗ Dataset directory not found")
        return False

if __name__ == "__main__":
    print("=== Testing Pipeline Requirements ===")
    
    all_good = True
    all_good &= test_packages()
    all_good &= test_import()
    all_good &= test_dataset()
    
    if all_good:
        print("\n✓ All tests passed! Pipeline should work.")
        print("Now running the advanced pipeline...")
        
        try:
            from advanced_to_tinyml_pipeline import main
            main()
        except Exception as e:
            print(f"✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")

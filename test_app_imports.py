"""
Test script to check all app imports
"""
import sys
import traceback

def test_import(module_name, from_module=None):
    """Test importing a module and report results"""
    try:
        if from_module:
            exec(f"from {from_module} import {module_name}")
            print(f"✅ Successfully imported {module_name} from {from_module}")
        else:
            exec(f"import {module_name}")
            print(f"✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing all app imports...")
    print("=" * 50)
    
    # Test basic imports
    basic_imports = [
        "streamlit",
        "cv2", 
        "numpy",
        "tempfile",
        "os",
        "pathlib",
        "time",
        "pandas",
        "plotly.express",
        "plotly.graph_objects"
    ]
    
    print("\n📦 Testing basic imports:")
    for module in basic_imports:
        test_import(module)
    
    # Test custom module imports
    print("\n🔧 Testing custom module imports:")
    custom_imports = [
        ("EnhancedFootballTracker", "tracking_system"),
        ("YOLO_MODELS", "config"),
        ("StreamlitVideoProcessor", "video_processor"),
        ("TacticalAnalyzer", "tactical_analyzer"),
        ("HawkEyeAnalyzer", "hawkeye_analyzer")
    ]
    
    for class_name, module_name in custom_imports:
        test_import(class_name, module_name)
    
    # Test ultralytics specifically
    print("\n🎯 Testing YOLO/Ultralytics:")
    test_import("ultralytics")
    if test_import("YOLO", "ultralytics"):
        try:
            from ultralytics import YOLO
            print("✅ YOLO class imported successfully")
        except Exception as e:
            print(f"❌ YOLO class import failed: {str(e)}")
    
    print("\n🏁 Import testing completed!")

if __name__ == "__main__":
    main()

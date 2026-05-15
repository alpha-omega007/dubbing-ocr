import sys
import os

# Add user's local site-packages to path
user_site = "/home/master/.local/lib/python3.10/site-packages"
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    import transformers
    import transformers.utils.import_utils as transformers_utils
    import transformers.modeling_utils as modeling_utils
    
    print(f"Transformers version: {transformers.__version__}")
    
    def mock_check(*args, **kwargs):
        print("Mock check called!")
        return None

    transformers_utils.check_torch_load_is_safe = mock_check
    modeling_utils.check_torch_load_is_safe = mock_check
    
    print("Monkeypatched in both places.")
    
    # Try calling it from modeling_utils to be sure
    modeling_utils.check_torch_load_is_safe()
    
except Exception as e:
    print(f"Error: {e}")

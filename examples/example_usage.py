"""Example usage of the new BackendManager with provider pattern.

This script demonstrates how to use the refactored BackendManager 
and how to add new providers.
"""

from sudoku_nisq.backends import BackendManager
from sudoku_nisq.providers.aws import AWSProvider


def main():
    """Demonstrate the new BackendManager usage."""
    
    # Create manager instance
    manager = BackendManager()
    
    print("=== BackendManager with Provider Pattern ===")
    print(f"Available providers: {manager.list_providers()}")
    print()
    
    # Example 1: Using IBM provider
    print("=== IBM Provider Example ===")
    try:
        # Note: This will fail without real credentials
        # devices = manager.authenticate_ibm(
        #     api_token="your_token_here",
        #     instance="your_instance_here"
        # )
        # manager.add_backend("ibm", "ibm_brisbane", alias="my_ibm_device")
        print("IBM authentication and device setup would happen here")
        print("(Skipping due to missing credentials)")
    except Exception as e:
        print(f"IBM setup failed (expected): {e}")
    print()
    
    # Example 2: Using Quantinuum provider
    print("=== Quantinuum Provider Example ===")
    try:
        # Note: This will fail without real credentials
        # devices = manager.authenticate_quantinuum()
        # manager.add_backend("quantinuum", "H1-1", alias="my_quantinuum_device")
        print("Quantinuum authentication and device setup would happen here")
        print("(Skipping due to missing credentials)")
    except Exception as e:
        print(f"Quantinuum setup failed (expected): {e}")
    print()
    
    # Example 3: Adding a new provider (AWS)
    print("=== Adding New Provider (AWS) ===")
    aws_provider = AWSProvider()
    manager.register_provider(aws_provider)
    print(f"Providers after adding AWS: {manager.list_providers()}")
    
    # Authenticate with AWS (placeholder)
    devices = manager.get_provider("aws").authenticate()
    print(f"Available AWS devices: {devices}")
    
    # Add AWS device
    manager.add_backend("aws", devices[0], alias="my_aws_device")
    print(f"Registered backends: {manager.all()}")
    print()
    
    # Example 4: Unified backend access
    print("=== Unified Backend Access ===")
    try:
        aws_backend = manager.get("my_aws_device")
        print(f"AWS backend: {aws_backend}")
        
        # Show backend info
        info = manager.info()
        print(f"Backend info: {info}")
        
    except Exception as e:
        print(f"Backend access failed: {e}")
    print()
    
    # Example 5: Provider-specific methods still available
    print("=== Provider-specific Access ===")
    aws_provider = manager.get_provider("aws")
    print(f"AWS provider: {aws_provider.provider_name}")
    print(f"AWS backends: {aws_provider.list_backends()}")
    print()
    
    # Example 6: Easy backend management
    print("=== Backend Management ===")
    print(f"Total backends: {manager.count()}")
    print(f"Is 'my_aws_device' registered? {manager.is_registered('my_aws_device')}")
    
    # Clean up
    manager.clear()
    print(f"Backends after clear: {manager.all()}")


if __name__ == "__main__":
    main()
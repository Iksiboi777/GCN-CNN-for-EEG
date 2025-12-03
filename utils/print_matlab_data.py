# Recursively print the structure of nested objects
def print_structure(obj, indent=0, name="root"):
    prefix = " " * indent
    if isinstance(obj, dict):
        print(f"{prefix}{name} (dict):")
        for k, v in obj.items():
            print_structure(v, indent + 2, k)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name} ({type(obj).__name__}, len={len(obj)}):")
        for i, v in enumerate(obj):
            print_structure(v, indent + 2, f"[{i}]")
    elif hasattr(obj, 'dtype') and obj.dtype.names:  # numpy structured array
        print(f"{prefix}{name} (np.structured, fields={obj.dtype.names}):")
        for field in obj.dtype.names:
            print_structure(obj[field], indent + 2, field)
    elif hasattr(obj, 'shape'):
        print(f"{prefix}{name} (np.ndarray, shape={obj.shape}, dtype={obj.dtype})")
        # Optionally, print a sample value
    else:
        print(f"{prefix}{name}: {type(obj).__name__}")



def print_full_tree(obj, indent=0, name="root", max_depth=20):
    prefix = " " * indent
    if indent // 2 > max_depth:
        print(f"{prefix}{name}: ...max depth reached...")
        return
    # Print type and shape/info
    if isinstance(obj, dict):
        print(f"{prefix}{name} (dict):")
        for k, v in obj.items():
            print_full_tree(v, indent + 2, k, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name} ({type(obj).__name__}, len={len(obj)}):")
        for i, v in enumerate(obj):
            print_full_tree(v, indent + 2, f"[{i}]", max_depth)
    elif hasattr(obj, 'dtype') and obj.dtype.names:  # numpy structured array
        print(f"{prefix}{name} (np.structured, fields={obj.dtype.names}):")
        for field in obj.dtype.names:
            print_full_tree(obj[field], indent + 2, field, max_depth)
    elif hasattr(obj, 'shape') and obj.dtype == 'O':  # object array
        print(f"{prefix}{name} (np.ndarray, shape={obj.shape}, dtype=object):")
        it = obj.flat if obj.ndim > 1 else obj
        for idx, v in enumerate(it):
            print_full_tree(v, indent + 2, f"[{idx}]", max_depth)
    elif hasattr(obj, 'shape'):
        print(f"{prefix}{name} (np.ndarray, shape={obj.shape}, dtype={obj.dtype})")
    else:
        print(f"{prefix}{name}: {type(obj).__name__} - {repr(obj)[:60]}")

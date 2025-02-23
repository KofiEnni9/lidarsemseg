"""
Data Cache Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
import numpy as np
from multiprocessing import shared_memory

def shared_array(name, var=None):
    if var is not None:
        # Check if shared memory already exists
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_array = np.ndarray(var.shape, dtype=var.dtype, buffer=existing_shm.buf)
            return existing_array
        except FileNotFoundError:
            # Create new shared memory
            shm = shared_memory.SharedMemory(name=name, create=True, size=var.nbytes)
            shared_array = np.ndarray(var.shape, dtype=var.dtype, buffer=shm.buf)
            shared_array[...] = var[...]
            return shared_array
    else:
        # Attach to existing shared memory
        shm = shared_memory.SharedMemory(name=name)
        return np.ndarray(var.shape, dtype=var.dtype, buffer=shm.buf)

def shared_dict(name, var=None):
    name = str(name)
    assert "." not in name  # '.' is used as sep flag
    data = {}
    
    if var is not None:
        assert isinstance(var, dict)
        keys = [key for key in var.keys() if isinstance(var[key], np.ndarray)]
        
        # Store keys in a separate shared memory segment
        keys_shm = shared_memory.SharedMemory(name=f"{name}.keys", create=True, size=len(keys) * 100)
        keys_array = np.frombuffer(keys_shm.buf, dtype='<U100')
        keys_array[:len(keys)] = keys
        
        for key in keys:
            if isinstance(var[key], np.ndarray):
                data[key] = shared_array(name=f"{name}.{key}", var=var[key])
    else:
        # Retrieve keys from shared memory
        keys_shm = shared_memory.SharedMemory(name=f"{name}.keys")
        keys_array = np.frombuffer(keys_shm.buf, dtype='<U100')
        keys = [key for key in keys_array if key]
        
        for key in keys:
            data[key] = shared_array(name=f"{name}.{key}")
    
    return data
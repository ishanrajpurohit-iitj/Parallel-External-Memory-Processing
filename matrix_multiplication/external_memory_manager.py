import mmap

def create_memory_mapped_file(filename, size):
    try:
        with open(filename, "wb") as f:
            f.write(b"\x00" * size)
        
        with mmap.mmap(-1, size, access=mmap.ACCESS_WRITE) as mm:
            # Use the memory-mapped file here
            # For example, to write data to the first 10 bytes:
            mm[:10] = b"Hello, world!"
    except Exception as e:
        print(f"Error creating or accessing memory-mapped file: {e}")

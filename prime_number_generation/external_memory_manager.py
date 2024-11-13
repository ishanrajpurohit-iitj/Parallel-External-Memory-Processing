# External memory manager example for large data handling
import mmap

def create_memory_mapped_file(filename, size):
    with open(filename, "wb") as f:
        f.write(b"\x00" * size)
    return mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)

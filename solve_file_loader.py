import numpy as np
import numpy.typing as npt
import struct

def read_8byte_array(filename: str) -> tuple[int, npt.NDArray[np.uint64]]:
    '''
    reads a file from disk.
    format of file:
    - 16 bytes: header
        - 8 bytes: number of entries
        - 8 bytes: memory size
    - [memory size] * 8 bytes: the actual data to load
    Based on NBitArray<numBits>::Read(FILE *f) in NBitArray.h
    '''
    with open(filename, 'rb') as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("File too short to contain header.")

        entries, memory_size = struct.unpack('QQ', header)

        # Read packed memory data
        data_bytes = f.read(memory_size * 8)
        if len(data_bytes) != memory_size * 8:
            raise ValueError("File too short for expected memory content.")

        mem = np.frombuffer(data_bytes, dtype=np.uint64)

    return entries, mem

def get_value_2(mem: npt.NDArray[np.uint64], index: int) -> int:
    '''
    gets the value corresponding to the given index in mem,
    assuming 2-bit packing in the uint64 elements.
    based on NBitArray<2>::Get in NBitArray.cpp
    '''
    word_index = index >> 5 # index // 32
    bit_offset = (index & 0x1F) << 1 # (index % 32) * 2
    return (mem[word_index] >> bit_offset) & 0x3

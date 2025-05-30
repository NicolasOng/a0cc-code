import numpy as np
import numpy.typing as npt
import struct
from enum import Enum

class Outcome(Enum):
    WIN = 2
    LOSS = 1
    DRAW = 0
    ILLEGAL = 3

class SolveData:
    def __init__(self, filename: str):
        self.entries, self.mem = self.read_solve_data_file(filename)
    
    @staticmethod
    def read_solve_data_file(filename: str) -> tuple[int, npt.NDArray[np.uint64]]:
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

            # Use little-endian for unpacking
            entries, memory_size = struct.unpack('<QQ', header)

            # Read packed memory data
            data_bytes = f.read(memory_size * 8)
            if len(data_bytes) != memory_size * 8:
                raise ValueError("File too short for expected memory content.")

            # Use little-endian for numpy array
            mem = np.frombuffer(data_bytes, dtype='<u8')

        return entries, mem
    
    @staticmethod
    def get_value_2(mem: npt.NDArray[np.uint64], index: int) -> int:
        '''
        gets the value corresponding to the given index in mem,
        assuming 2-bit packing in the uint64 elements.
        based on NBitArray<2>::Get in NBitArray.cpp
        '''
        word_index = index >> 5 # index // 32
        bit_offset = (index & 0x1F) << 1 # (index % 32) * 2
        return (mem[word_index] >> bit_offset) & 0x3
    
    def get(self, index: int) -> int:
        '''
        gets the value corresponding to the given index in mem,
        assuming 2-bit packing in the uint64 elements.
        '''
        return self.get_value_2(self.mem, index)

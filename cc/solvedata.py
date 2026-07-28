import numpy as np
import numpy.typing as npt
import os
import struct
from enum import Enum

class Outcome(Enum):
    WIN = 2
    LOSS = 1
    DRAW = 0
    ILLEGAL = 3

class SolveData:
    def __init__(self, filename: str, mmap: bool = True):
        self.entries, self.mem = self.read_solve_data_file(filename, mmap)

    @staticmethod
    def _resolve_staged_path(filename: str) -> str:
        '''If a copy of this solve file has been staged into $SLURM_TMPDIR
        (node-local NVMe; see slurm/stage_solve_data.sh), read that instead of
        the original path so mmap page faults hit local disk rather than the
        network filesystem. Off-cluster ($SLURM_TMPDIR unset) this is a no-op and
        the original path is mmapped in place.'''
        tmp = os.environ.get('SLURM_TMPDIR')
        if tmp:
            staged = os.path.join(tmp, os.path.basename(filename))
            if os.path.exists(staged):
                return staged
        return filename

    @staticmethod
    def read_solve_data_file(filename: str, mmap: bool = True) -> tuple[int, npt.NDArray[np.uint64]]:
        '''
        reads a file from disk.
        format of file:
        - 16 bytes: header
            - 8 bytes: number of entries
            - 8 bytes: memory size
        - [memory size] * 8 bytes: the actual data to load
        Based on NBitArray<numBits>::Read(FILE *f) in NBitArray.h
        '''
        filename = SolveData._resolve_staged_path(filename)
        with open(filename, 'rb') as f:
            header = f.read(16)
            if len(header) != 16:
                raise ValueError("File too short to contain header.")

            # Use little-endian for unpacking
            entries, memory_size = struct.unpack('<QQ', header)

        if mmap:
            mem = np.memmap(filename, dtype='<u8', mode='r', offset=16, shape=(memory_size,))
        else:
            with open(filename, 'rb') as f:
                f.seek(16)
                data_bytes = f.read(memory_size * 8)
                if len(data_bytes) != memory_size * 8:
                    raise ValueError("File too short for expected memory content.")
                mem = np.frombuffer(data_bytes, dtype='<u8')

        return entries, mem
    
    @staticmethod
    def get_value_2(mem: npt.NDArray[np.uint64], index: int) -> int:
        '''
        gets the value corresponding to the given index in mem,
        assuming 2-bit packing in the uint64 elements.
        based on NBitArray<2>::Get in NBitArray.cpp
        '''
        index = int(index)
        word_index = index >> 5 # index // 32
        bit_offset = (index & 0x1F) << 1 # (index % 32) * 2
        return (int(mem[word_index]) >> bit_offset) & 0x3
    
    def get(self, index: int) -> int:
        '''
        gets the value corresponding to the given index in mem,
        assuming 2-bit packing in the uint64 elements.
        '''
        return self.get_value_2(self.mem, index)

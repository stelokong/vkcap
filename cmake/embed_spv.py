#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input.spv> <output.hpp> <symbol_name>")
    in_path, out_path, sym = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(in_path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        sys.exit(f"Invalid SPIR-V size {len(data)} (not multiple of 4)")
    words = [int.from_bytes(data[i:i+4], "little") for i in range(0, len(data), 4)]
    with open(out_path, "w") as out:
        out.write("#pragma once\n#include <cstdint>\n#include <array>\n\n")
        out.write(f"inline constexpr std::array<uint32_t, {len(words)}> {sym} = {{\n")
        for i in range(0, len(words), 8):
            chunk = ", ".join(f"0x{w:08x}" for w in words[i:i+8])
            out.write(f"    {chunk},\n")
        out.write("};\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import csv
import subprocess
from datetime import datetime

# sweep parameters
TILE_SIZES = [8, 16, 32, 64]
THREAD_TILES = [4, 8]
SIZES = [4096, 8192] 
# [64, 128, 256, 512, 1024, 2048] #, ]

TEMPLATE = "tgemm_main.cu"
WORKDIR = "."
OUT_CSV = "autotune_results.csv"
MAKE_CMD = ["make"]      # will append TILE=.. THREAD_TILE=..
RUN_TIMEOUT = 3        # seconds per run

os.makedirs(WORKDIR, exist_ok=True)

def is_valid_combo(tile, ttile):
    if tile % ttile != 0:
        return False
    if ttile != 1 and (ttile % 4 != 0):
        return False
    bx = tile // ttile
    by = tile // ttile
    if bx * by > 1024:
        return False
    return True

def sed_generate(template_path, out_path, tile, ttile):
    print(template_path)
    sed_cmd = [
        "sed", 
        "-i",
        "-e", f"s/const uint tile_size = .*/const uint tile_size = {tile};/g",
        "-e", f"s/const uint thread_tile = .*/const uint thread_tile = {ttile};/g",
        template_path
    ]
    subprocess.run(sed_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    # with open(out_path, "w", encoding="utf-8") as out_f:
    #     subprocess.run(sed_cmd, stdout=out_f, stderr=subprocess.DEVNULL, check=True)

def run_make(tile, ttile):
    cmd = MAKE_CMD + [f"TILE={tile}", f"THREAD_TILE={ttile}"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

def run_binary(exe_path, size_arg):
    if not os.path.isfile(exe_path):
        return 127, f"missing binary: {exe_path}"
    try:
        p = subprocess.run([exe_path, str(size_arg), str(1)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=RUN_TIMEOUT)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        return 124, str(e.stdout)
    except Exception as e:
        return 1, f"run error: {type(e).__name__}: {e}"

with open(OUT_CSV, "w", newline="", encoding="utf-8") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["timestamp_utc", "size", "tile_size", "thread_tile", "binary_name", "stdout"])
    for tile in TILE_SIZES:
        for ttile in THREAD_TILES:
            if not is_valid_combo(tile, ttile):
                continue

            base_no_size = f"tgemm_main"
            cu_out = os.path.join(WORKDIR, base_no_size + ".cu")

            sed_generate(TEMPLATE, cu_out, tile, ttile)

            cmd = MAKE_CMD
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

            for size in SIZES:
                if tile > size:
                    continue

                binary_name = f"tgemm"
                exe_path = os.path.join(".", binary_name)
                print(exe_path)
                rc, out = run_binary(exe_path, size)

                writer.writerow([datetime.utcnow().isoformat(), size, tile, ttile, binary_name, out.replace("\r", "")])
            
            subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                
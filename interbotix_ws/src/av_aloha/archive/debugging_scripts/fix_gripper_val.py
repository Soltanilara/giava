#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def replace_in_list_column(table, col_name, index_in_list, old_value, new_value, tol):
    col_idx = table.schema.get_field_index(col_name)
    if col_idx == -1:
        raise ValueError(f"Column not found: {col_name}")

    arr = table.column(col_idx)
    pylist = arr.to_pylist()
    changed = 0

    for row in pylist:
        if row is None:
            continue
        if index_in_list >= len(row):
            continue
        val = row[index_in_list]
        if val is None:
            continue
        if abs(float(val) - old_value) <= tol:
            row[index_in_list] = float(new_value)
            changed += 1

    new_arr = pa.array(pylist, type=arr.type)
    table = table.set_column(col_idx, col_name, new_arr)
    return table, changed


def process_parquet_file(path, old_value, new_value, tol, dry_run):
    table = pq.read_table(path)
    total_changed = 0

    for col_name in ["action", "observation.state"]:
        try:
            table, changed = replace_in_list_column(
                table,
                col_name=col_name,
                index_in_list=6,
                old_value=old_value,
                new_value=new_value,
                tol=tol,
            )
            total_changed += changed
        except ValueError:
            pass

    if total_changed > 0 and not dry_run:
        pq.write_table(table, path)

    return total_changed


def main():
    parser = argparse.ArgumentParser(description="Replace LeRobot gripper values in parquet files.")
    parser.add_argument("dataset_root", type=str, nargs="?", default="/home/devi/giava/interbotix_ws/src/av_aloha/data_collection_scripts/dataset/lerobot/grasp_cube/20260529_162433", help="Path to LeRobot dataset root")
    parser.add_argument("--old", type=float, default=-1.5, help="Old gripper value to replace")
    parser.add_argument("--new", type=float, default=1.0, help="New gripper value")
    parser.add_argument("--tol", type=float, default=1e-5, help="Tolerance for float comparison")
    parser.add_argument("--backup", action="store_true", help="Create a backup copy of the dataset root before editing")
    parser.add_argument("--dry-run", action="store_true", help="Report matches without writing changes")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if args.backup and not args.dry_run:
        backup_root = dataset_root.parent / f"{dataset_root.name}_backup_before_gripper_fix"
        if backup_root.exists():
            raise FileExistsError(f"Backup path already exists: {backup_root}")
        shutil.copytree(dataset_root, backup_root)
        print(f"Created backup: {backup_root}")

    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root}")

    grand_total = 0
    for path in parquet_files:
        changed = process_parquet_file(path, args.old, args.new, args.tol, args.dry_run)
        if changed > 0:
            print(f"{path}: changed {changed} values")
            grand_total += changed

    mode = "Would change" if args.dry_run else "Changed"
    print(f"{mode} {grand_total} total values from {args.old} to {args.new}")


if __name__ == "__main__":
    main()
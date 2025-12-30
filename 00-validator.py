#!/usr/bin/env python3

from pathlib import Path
import json
import argparse

from rich.console import Console
from rich.table import Table

from src.validate_exports import validate_directory

console = Console()


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate TissueFAXS OME-TIFF exports before processing."
    )
    p.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing *_channels.tiff files. Export should be done to single tiff with n channels.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to write validation_report.json (default: <input_dir>/validation_report.json)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    report_path = (
        args.report.resolve()
        if args.report is not None
        else input_dir / "validation_report.json"
    )

    console.rule("[bold]Stage 00 · Validation[/bold]")
    console.print(f"[bold]Input directory:[/bold] {input_dir}")

    results = validate_directory(input_dir)

    # overwrite report unconditionally
    report_path.unlink(missing_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    table = Table(title="Validation summary")
    table.add_column("Region", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Warnings", justify="right")
    table.add_column("Errors", justify="right")

    failed = False
    for region, res in results.items():
        ok = res["ok"]
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(
            region,
            status,
            str(len(res["warnings"])),
            str(len(res["errors"])),
        )
        if not ok:
            failed = True

    console.print(table)
    console.print(f"[bold]Report written to:[/bold] {report_path}")

    if failed:
        console.rule("[red]Validation failed[/red]")
        raise SystemExit(1)

    console.rule("[green]Validation passed[/green]")


if __name__ == "__main__":
    main()

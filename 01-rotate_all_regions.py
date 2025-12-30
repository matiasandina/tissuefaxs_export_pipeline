#!/usr/bin/env python3

from pathlib import Path
import argparse

from rich.console import Console

from src.rotate_channels import rotate_channels_file

console = Console()


def parse_args():
    p = argparse.ArgumentParser(
        description="Rotate TissueFAXS multi-channel OME-TIFFs by 180°."
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing *_channels.tiff files",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory (default: overwrite in place)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    outdir = args.output_dir.resolve() if args.output_dir else None
    dry = args.dry_run

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]Stage 01 · Rotation[/bold]")
    console.print(f"[bold]Input directory:[/bold] {input_dir}")
    console.print(
        f"[bold]Mode:[/bold] {'dry-run' if dry else 'write'} · "
        f"{'overwrite in place' if outdir is None else f'write to {outdir}'}"
    )

    files = sorted(input_dir.glob("*_channels.tiff"))

    if not files:
        console.print("[red]No *_channels.tiff files found[/red]")
        raise SystemExit(1)

    for p in files:
        dst = outdir / p.name if outdir else p
        console.print(
            f"{'[yellow][DRY][/yellow] ' if dry else ''}"
            f"Rotate {p.name} → {dst}"
        )
        if not dry:
            rotate_channels_file(p, dst)

    console.rule("[green]Rotation complete[/green]")


if __name__ == "__main__":
    main()

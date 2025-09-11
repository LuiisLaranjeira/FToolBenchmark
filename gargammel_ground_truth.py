#!/usr/bin/env python3
"""
Gargammel Ground Truth Generator

This script generates ground truth files from a Gargammel folder structure.
It processes FASTA files in bact/, endo/, and cont/ folders and creates
taxonomic ground truth files by extracting accessions and mapping them to taxIDs.

Usage:
    python gargammel_ground_truth.py --data_dir data/ --email your@email.com [--api_key YOUR_KEY]

Expected folder structure:
    data/
    ├── bact/     (bacterial sequences - .fa/.fna/.fasta files)
    ├── endo/     (endogenous sequences)
    └── cont/     (contaminant sequences)
"""

import os
import sys
import argparse
import glob
import re

# Import functions from your existing taxonomy processing script
try:
    from main import (
        extract_accessions_from_fna,
        accession_to_taxid,
        generate_ground_truth_from_fna_folder
    )

    print("[✓] Successfully imported functions from taxonomy_processor.py")
except ImportError as e:
    print(f"[!] Could not import from taxonomy_processor.py: {e}")
    print("[!] Make sure the taxonomy processing script is in the same directory")
    print("[!] Using built-in functions instead...")


    # Fallback implementations
    def extract_accessions_from_fna(folder_path):
        """Extract accession numbers from the first line of each FASTA file."""
        accessions = []
        fasta_extensions = ['*.fna', '*.fa', '*.fasta', '*.fas']

        for ext in fasta_extensions:
            pattern = os.path.join(folder_path, ext)
            for filepath in glob.glob(pattern):
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            if line.startswith(">"):
                                # Extract accession from header
                                acc = line.split()[0][1:]  # Remove '>' and take first part
                                accessions.append(acc)
                                break  # Only take first accession per file
                except Exception as e:
                    print(f"[!] Error reading {filepath}: {e}")
        return accessions


    def accession_to_taxid(accessions, email, api_key=None, sleep_time=0.34, verbose=False):
        """Placeholder - requires Bio.Entrez for actual implementation."""
        print(
            "[!] Warning: Using placeholder function. Install biopython and use original script for actual taxID mapping.")
        # Return dummy mapping for demonstration
        return {acc: "0" for acc in accessions}


def validate_gargammel_structure(data_dir):
    """
    Validate that the data directory has the expected Gargammel structure.

    Args:
        data_dir (str): Path to the data directory

    Returns:
        dict: Dictionary with folder paths and validation results
    """
    expected_folders = ['bact', 'endo', 'cont']
    structure = {}

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for folder in expected_folders:
        folder_path = os.path.join(data_dir, folder)
        structure[folder] = {
            'path': folder_path,
            'exists': os.path.exists(folder_path),
            'files': []
        }

        if structure[folder]['exists']:
            # Find FASTA files
            fasta_extensions = ['*.fna', '*.fa', '*.fasta', '*.fas']
            for ext in fasta_extensions:
                pattern = os.path.join(folder_path, ext)
                files = glob.glob(pattern)
                structure[folder]['files'].extend(files)

    return structure


def extract_accessions_from_file(filepath):
    """
    Extract accession numbers from a single FASTA file.

    Args:
        filepath (str): Path to the FASTA file

    Returns:
        list: List of accession numbers found in the file
    """
    accessions = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    # Extract accession from header line
                    # Handle different header formats
                    header = line.strip()[1:]  # Remove '>'

                    # Try different patterns for accession extraction
                    patterns = [
                        r'([A-Z]{1,2}_\d+\.\d+)',  # RefSeq format (e.g., NC_001925.1)
                        r'([A-Z]{1,2}\d+\.\d+)',  # GenBank format (e.g., AB123456.1)
                        r'([A-Z]+\d+)',  # Simple format (e.g., ABC123)
                        r'^(\S+)'  # First non-whitespace string
                    ]

                    accession_found = False
                    for pattern in patterns:
                        match = re.search(pattern, header)
                        if match:
                            accessions.append(match.group(1))
                            accession_found = True
                            break

                    if not accession_found:
                        # Fallback: take first word
                        first_word = header.split()[0] if header.split() else header
                        accessions.append(first_word)

    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")

    return accessions


def generate_combined_taxids_only(data_dir, output_file, email, api_key=None, verbose=False):
    """
    Generate a simple file with all unique taxonomic IDs from all FASTA files.

    Args:
        data_dir (str): Path to the data directory containing bact/, endo/, cont/
        output_file (str): Path to output file (e.g., "all_taxids.txt")
        email (str): Email for NCBI Entrez queries
        api_key (str, optional): NCBI API key for higher rate limits
        verbose (bool): Whether to print detailed output

    Returns:
        list: List of all unique taxonomic IDs
    """
    print(f"[▶] Generating combined taxIDs file from: {data_dir}")

    # Validate structure
    structure = validate_gargammel_structure(data_dir)

    # Collect all accessions from all categories
    all_accessions = []
    total_files = 0

    for category, info in structure.items():
        if not info['exists'] or not info['files']:
            continue

        print(f"[📁] Processing {category} ({len(info['files'])} files)...")

        for filepath in info['files']:
            accessions = extract_accessions_from_file(filepath)
            all_accessions.extend(accessions)
            total_files += 1

            if verbose:
                filename = os.path.basename(filepath)
                print(f"    {filename}: {len(accessions)} accessions")

    # Remove duplicates
    unique_accessions = list(set(all_accessions))
    print(f"[📊] Total files processed: {total_files}")
    print(f"[📊] Total accessions found: {len(all_accessions)}")
    print(f"[📊] Unique accessions: {len(unique_accessions)}")

    # Map to taxonomic IDs
    print(f"[🔍] Mapping {len(unique_accessions)} accessions to taxonomic IDs...")

    try:
        taxid_map = accession_to_taxid(
            unique_accessions,
            email=email,
            api_key=api_key,
            verbose=verbose
        )

        # Get unique taxIDs, excluding unmapped ones
        unique_taxids = set()
        for taxid in taxid_map.values():
            if taxid != "0":  # Skip unmapped accessions
                unique_taxids.add(taxid)

        # Sort taxIDs numerically
        sorted_taxids = sorted(unique_taxids, key=int)

        # Write to output file
        with open(output_file, 'w') as f:
            for taxid in sorted_taxids:
                f.write(f"{taxid}\n")

        print(f"[✅] Success! {len(sorted_taxids)} unique taxIDs written to: {output_file}")
        return sorted_taxids

    except Exception as e:
        print(f"[❌] Error mapping accessions to taxIDs: {e}")
        raise


def generate_ground_truth_gargammel(data_dir, output_dir, email, api_key=None,
                                    map_to_taxid=True, verbose=False):
    """
    Generate ground truth files from Gargammel folder structure.

    Args:
        data_dir (str): Path to the data directory containing bact/, endo/, cont/
        output_dir (str): Path to output directory for ground truth files
        email (str): Email for NCBI Entrez queries
        api_key (str, optional): NCBI API key for higher rate limits
        map_to_taxid (bool): Whether to map accessions to taxonomic IDs
        verbose (bool): Whether to print detailed output
    """
    print(f"[▶] Generating ground truth from Gargammel structure: {data_dir}")

    # Validate structure
    structure = validate_gargammel_structure(data_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each category
    results = {}

    for category, info in structure.items():
        print(f"\n[📁] Processing {category.upper()} category...")

        if not info['exists']:
            print(f"[⚠] Folder not found: {info['path']}")
            continue

        if not info['files']:
            print(f"[⚠] No FASTA files found in: {info['path']}")
            continue

        print(f"[📄] Found {len(info['files'])} FASTA files")

        # Extract accessions from all files in this category
        all_accessions = []
        file_details = {}

        for filepath in info['files']:
            filename = os.path.basename(filepath)
            if verbose:
                print(f"    Processing: {filename}")

            accessions = extract_accessions_from_file(filepath)
            all_accessions.extend(accessions)
            file_details[filename] = accessions

            if verbose:
                print(f"      Found {len(accessions)} accessions")

        # Remove duplicates and sort
        unique_accessions = sorted(list(set(all_accessions)))
        print(f"[✓] Total unique accessions in {category}: {len(unique_accessions)}")

        # Save accessions to file
        accessions_file = os.path.join(output_dir, f"ground_truth_{category}_accessions.txt")
        with open(accessions_file, 'w') as f:
            f.write(f"# Ground truth accessions for {category.upper()} category\n")
            f.write(f"# Generated from: {info['path']}\n")
            f.write(f"# Files processed: {len(info['files'])}\n")
            f.write(f"# Total accessions: {len(all_accessions)}\n")
            f.write(f"# Unique accessions: {len(unique_accessions)}\n")
            f.write(f"# Date: {os.popen('date').read().strip()}\n\n")

            for acc in unique_accessions:
                f.write(f"{acc}\n")

        print(f"[✓] Saved accessions to: {accessions_file}")

        # Map to taxonomic IDs if requested
        taxid_file = None
        taxid_map = {}
        if map_to_taxid and unique_accessions:
            print(f"[🔍] Mapping accessions to taxonomic IDs for {category}...")

            try:
                taxid_map = accession_to_taxid(
                    unique_accessions,
                    email=email,
                    api_key=api_key,
                    verbose=verbose
                )

                # Save taxID mapping
                taxid_file = os.path.join(output_dir, f"ground_truth_{category}_taxids.txt")
                with open(taxid_file, 'w') as f:
                    f.write(f"# Ground truth taxonomic IDs for {category.upper()} category\n")
                    f.write(f"# Generated from: {info['path']}\n")
                    f.write(f"# Mapped accessions: {len(taxid_map)}\n")
                    f.write(f"# Date: {os.popen('date').read().strip()}\n\n")

                    for acc, taxid in sorted(taxid_map.items()):
                        f.write(f"{acc}\t{taxid}\n")

                # Also save just the unique taxIDs
                unique_taxids = sorted(list(set(taxid_map.values())))
                taxids_only_file = os.path.join(output_dir, f"ground_truth_{category}_taxids_only.txt")
                with open(taxids_only_file, 'w') as f:
                    f.write(f"# Unique taxonomic IDs for {category.upper()} category\n")
                    f.write(f"# Total unique taxIDs: {len(unique_taxids)}\n\n")
                    for taxid in unique_taxids:
                        if taxid != "0":  # Skip unmapped accessions
                            f.write(f"{taxid}\n")

                print(f"[✓] Saved taxID mapping to: {taxid_file}")
                print(f"[✓] Saved unique taxIDs to: {taxids_only_file}")

            except Exception as e:
                print(f"[!] Error mapping accessions to taxIDs: {e}")
                print(f"[!] Make sure you have biopython installed and valid NCBI credentials")

        # Store results
        results[category] = {
            'files_processed': len(info['files']),
            'total_accessions': len(all_accessions),
            'unique_accessions': len(unique_accessions),
            'accessions_file': accessions_file,
            'taxid_file': taxid_file,
            'file_details': file_details,
            'taxid_map': taxid_map
        }

    # Generate combined taxIDs file from all categories
    if map_to_taxid:
        print(f"\n[🔄] Generating combined taxIDs file...")
        all_taxids = set()

        for category, result in results.items():
            if result['taxid_map']:
                for taxid in result['taxid_map'].values():
                    if taxid != "0":  # Skip unmapped accessions
                        all_taxids.add(taxid)

        # Save combined taxIDs file
        combined_taxids_file = os.path.join(output_dir, "ground_truth_all_taxids.txt")
        with open(combined_taxids_file, 'w') as f:
            for taxid in sorted(all_taxids, key=int):
                f.write(f"{taxid}\n")

        print(f"[✓] Combined taxIDs file saved to: {combined_taxids_file}")
        print(f"[📊] Total unique taxIDs across all categories: {len(all_taxids)}")

    # Generate summary report
    summary_file = os.path.join(output_dir, "ground_truth_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("GARGAMMEL GROUND TRUTH GENERATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Date generated: {os.popen('date').read().strip()}\n\n")

        total_files = 0
        total_accessions = 0
        total_unique = 0

        for category, result in results.items():
            f.write(f"{category.upper()} CATEGORY:\n")
            f.write(f"  Files processed: {result['files_processed']}\n")
            f.write(f"  Total accessions: {result['total_accessions']}\n")
            f.write(f"  Unique accessions: {result['unique_accessions']}\n")
            f.write(f"  Accessions file: {os.path.basename(result['accessions_file'])}\n")
            if result['taxid_file']:
                f.write(f"  TaxID file: {os.path.basename(result['taxid_file'])}\n")
            f.write("\n")

            total_files += result['files_processed']
            total_accessions += result['total_accessions']
            total_unique += result['unique_accessions']

        f.write(f"TOTALS:\n")
        f.write(f"  Files processed: {total_files}\n")
        f.write(f"  Total accessions: {total_accessions}\n")
        f.write(f"  Total unique accessions: {total_unique}\n")

    print(f"\n[✅] Ground truth generation completed!")
    print(f"[📊] Summary saved to: {summary_file}")
    print(f"[📁] All files saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth files from Gargammel folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected folder structure:
    data/
    ├── bact/     (bacterial sequences - .fa/.fna/.fasta files)
    ├── endo/     (endogenous sequences)
    └── cont/     (contaminant sequences)

Example usage:
    python gargammel_ground_truth.py --data_dir data/ --email your@email.com
    python gargammel_ground_truth.py --data_dir data/ --email your@email.com --api_key YOUR_KEY --no_taxid
        """
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to data directory containing bact/, endo/, cont/ folders"
    )

    parser.add_argument(
        "--output_dir",
        default="ground_truth_output",
        help="Output directory for ground truth files (default: ground_truth_output)"
    )

    parser.add_argument(
        "--email",
        required=True,
        help="Email address for NCBI Entrez queries (required)"
    )

    parser.add_argument(
        "--api_key",
        default=None,
        help="NCBI API key for higher rate limits (optional)"
    )

    parser.add_argument(
        "--no_taxid",
        action="store_true",
        help="Skip mapping accessions to taxonomic IDs (faster, accessions only)"
    )

    parser.add_argument(
        "--combined_only",
        action="store_true",
        help="Generate only the combined taxIDs file (faster, single output file)"
    )

    parser.add_argument(
        "--output_file",
        default=None,
        help="Output file path for combined taxIDs (used with --combined_only)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output during processing"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.data_dir):
        print(f"[❌] Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Handle combined-only mode
    if args.combined_only:
        output_file = args.output_file or os.path.join(args.output_dir, "all_taxids.txt")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            taxids = generate_combined_taxids_only(
                data_dir=args.data_dir,
                output_file=output_file,
                email=args.email,
                api_key=args.api_key,
                verbose=args.verbose
            )

            print(f"\n[🎉] Success! Generated combined taxIDs file with {len(taxids)} unique taxIDs.")
            print(f"[📄] File: {output_file}")

        except Exception as e:
            print(f"[❌] Error: {e}")
            sys.exit(1)

        return

    # Generate ground truth
    try:
        results = generate_ground_truth_gargammel(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            email=args.email,
            api_key=args.api_key,
            map_to_taxid=not args.no_taxid,
            verbose=args.verbose
        )

        print("\n[🎉] Success! Ground truth files generated.")
        print("\nNext steps:")
        print("1. Use the generated files to validate your taxonomy classifiers")
        print("2. Compare classifier outputs against these ground truth files")
        print("3. Calculate precision, recall, and F1 scores for performance evaluation")

    except Exception as e:
        print(f"[❌] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
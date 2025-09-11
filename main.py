import csv
import os
import re
import time
from logging import exception

import pandas as pd
import argparse
from collections import Counter
from time import sleep
from glob import glob
from Bio import Entrez

def reads_per_taxon_centrifuge(input_file, output_file, empty_namesranks, empty_correct_incorrect):
    # Step 1: Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    taxid_list = []

    # Step 2: Read column 3 from input (skip header with 'taxID')
    with open(input_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3 or fields[2] == "taxID":
                continue
            taxid_list.append(fields[2])

    # Step 3: Count occurrences
    taxid_counts = Counter(taxid_list)

    # Step 4: Write to output
    with open(output_file, 'w') as out:
        for taxid, count in sorted(taxid_counts.items()):
            out.write(f"{count}\t{taxid}\n")

    # Step 5: If output is empty, write default files
    if os.path.getsize(output_file) == 0:
        with open(empty_namesranks, 'w') as f1:
            f1.write("0\t0\t0\t0\t0\n")
        with open(empty_correct_incorrect, 'w') as f2:
            f2.write("0\t0\t0\t0\t0\tUnclassified\t0\n")

def reads_per_taxon_kraken2(input_file, output_file, empty_namesranks, empty_correct_incorrect):
    taxids = []

    try:
        with open(input_file, 'r') as f:
            for line in f:
                if not line.startswith('C'):
                    continue  # Skip unclassified reads

                fields = line.strip().split('\t')
                if len(fields) < 3:
                    continue

                taxid = fields[2]
                if taxid.isdigit():
                    taxids.append(taxid)

        # Count taxID frequencies
        taxid_counts = Counter(taxids)

        # Write output file
        with open(output_file, 'w') as out:
            for taxid in sorted(taxid_counts, key=int):
                out.write(f"{taxid_counts[taxid]}\t{taxid}\n")

        # If output file is empty, write fallback files
        if os.path.getsize(output_file) == 0:
            raise ValueError("Empty classification output")

    except Exception:
        # Write empty placeholder files on failure or no classified reads
        with open(empty_namesranks, 'w') as f1:
            f1.write("0\t0\t0\t0\t0\n")
        with open(empty_correct_incorrect, 'w') as f2:
            f2.write("0\t0\t0\t0\t0\tUnclassified\t0\n")

def reads_per_taxon_diamond(input_file, output_file, empty_namesranks, empty_correct_incorrect):
    taxid_list = []

    try:
        # Step 1: Read file and extract column 2 (taxID), excluding '0'
        with open(input_file, 'r') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) >= 2 and fields[1] != '0':
                    taxid_list.append(fields[1])

        # Step 2: Count occurrences
        taxid_counts = Counter(taxid_list)

        # Step 3: Write results to output
        with open(output_file, 'w') as out:
            for taxid in sorted(taxid_counts, key=int):
                out.write(f"{taxid_counts[taxid]}\t{taxid}\n")

        # Step 4: Check if output is empty
        if os.path.getsize(output_file) == 0:
            raise ValueError("Empty result file")

    except Exception:
        # Step 5: Write fallback empty files
        with open(empty_namesranks, 'w') as f1:
            f1.write("0\t0\t0\t0\t0\n")
        with open(empty_correct_incorrect, 'w') as f2:
            f2.write("0\t0\t0\t0\t0\tUnclassified\t0\n")

def reads_per_taxon_metaphlan2(input_file, output_counts, output_names, empty_namesranks, empty_correct_incorrect):
    lineages = []

    try:
        # Read and clean input
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 2:
                    continue
                lineage = re.sub(r'\|t__.+$', '', fields[1])  # Remove terminal strain-level info
                lineages.append(lineage)

        # Count occurrences of each lineage
        lineage_counts = Counter(lineages)

        # Write counts file (only counts)
        with open(output_counts, 'w') as f_counts:
            for lineage in sorted(lineage_counts):
                f_counts.write(f"{lineage_counts[lineage]}\n")

        # Write last taxonomic rank name (e.g., species)
        with open(output_names, 'w') as f_names:
            for lineage in sorted(lineage_counts):
                match = re.search(r'\|(\w__)?.+?$', lineage)
                if match:
                    last_rank = match.group(0).lstrip('|')
                    last_rank = re.sub(r'^\w__', '', last_rank)
                    f_names.write(last_rank.replace('_', ' ') + '\n')

        # Check if output file is empty
        if os.path.getsize(output_counts) == 0:
            raise ValueError("Output is empty")

    except Exception:
        # Create fallback files
        with open(empty_namesranks, 'w') as f1:
            f1.write("0\t0\t0\t0\t0\n")
        with open(empty_correct_incorrect, 'w') as f2:
            f2.write("0\t0\t0\t0\t0\tUnclassified\t0\n")
        open(output_names, 'a').close()  # Touch the names file

def reads_per_taxon_clark(input_file, output_file, empty_namesranks, empty_correct_incorrect):
    # Step 1: Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    taxid_list = []
    try:

        # Step 2: Read TaxID and Count columns, skip 'UNKNOWN'
        with open(input_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                taxid = row.get("TaxID")
                count_str = row.get("Count")

                if taxid == "UNKNOWN":
                    continue

                try:
                    count = int(count_str)
                    taxid_list.extend([taxid] * count)
                except (ValueError, TypeError):
                    continue  # Skip malformed entries

        # Step 3: Count occurrences
        taxid_counts = Counter(taxid_list)

        # Step 4: Write to output
        with open(output_file, 'w') as out:
            for taxid, count in sorted(taxid_counts.items(), key=lambda x: int(x[0])):
                out.write(f"{count}\t{taxid}\n")

    # Exception handling for empty output
    except Exception:
        # Step 5: Write default fallback files if no data
        with open(empty_namesranks, 'w') as f1:
            f1.write("0\t0\t0\t0\t0\n")
        with open(empty_correct_incorrect, 'w') as f2:
            f2.write("0\t0\t0\t0\t0\tUnclassified\t0\n")

def reads_per_taxon_kaiju(input_file, output_file, empty_namesranks, empty_correct_incorrect):
    # Step 1: Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    taxid_list = []

    # Step 2: Read file and extract taxIDs from classified reads
    with open(input_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue
            if fields[0] != 'C':
                continue  # Only include classified reads
            taxid = fields[2]
            if taxid == '0':
                continue  # Skip unclassified
            taxid_list.append(taxid)

    # Step 3: Count occurrences
    taxid_counts = Counter(taxid_list)

    # Step 4: Write output
    with open(output_file, 'w') as out:
        for taxid, count in sorted(taxid_counts.items(), key=lambda x: int(x[0])):
            out.write(f"{count}\t{taxid}\n")

    # Step 5: If output is empty, write fallback files
    if os.path.getsize(output_file) == 0:
        with open(empty_namesranks, 'w') as f1:
            f1.write("0\t0\t0\t0\t0\n")
        with open(empty_correct_incorrect, 'w') as f2:
            f2.write("0\t0\t0\t0\t0\tUnclassified\t0\n")

def reads_per_taxon_falcon(input_file, output_file, empty_namesranks, empty_correct_incorrect):
    # Step 1: Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    taxon_counts = Counter()

    # Step 2: Read file and extract similarity and accession
    with open(input_file, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            similarity_str = row.get("Similarity")
            sequence = row.get("Sequence")

            if not similarity_str or not sequence:
                continue

            # Extract accession using regex
            match = re.search(r"[A-Z]{1,2}_\d+\.\d+", sequence)
            if not match:
                continue
            accession = match.group(0)

            try:
                similarity = float(similarity_str)
                taxon_counts[accession] += similarity
            except ValueError:
                continue

    # Step 3: Write output
    with open(output_file, 'w') as out:
        for taxid, similarity in sorted(taxon_counts.items(), key=lambda x: x[0]):
            out.write(f"{similarity:.3f}\t{taxid}\n")

    # Step 4: Fallback if output is empty
    if os.path.getsize(output_file) == 0:
        with open(empty_namesranks, 'w') as f1:
            f1.write("0\t0\t0\t0\t0\n")
        with open(empty_correct_incorrect, 'w') as f2:
            f2.write("0\t0\t0\t0\t0\tUnclassified\t0\n")

def get_name_rank(input_file, name_rank_file, joined_output_file, email, api_key=None):
    # Step 1: Set Entrez credentials
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # Step 2: Read taxIDs from input file (2nd column)
    taxids = []
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    with open(input_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                taxids.append(fields[1])

    # Step 3: Query NCBI for taxonomic info in batches
    tax_info = {}
    batch_size = 500

    for i in range(0, len(taxids), batch_size):
        batch = taxids[i:i + batch_size]
        try:
            handle = Entrez.efetch(db="taxonomy", id=",".join(batch), retmode="xml")
            records = Entrez.read(handle)
            for record in records:
                tid = record.get("TaxId", "NA")
                name = record.get("ScientificName", "NA")
                rank = record.get("Rank", "NA")
                tax_info[tid] = (name, rank)
            sleep(0.34)  # rate limiting
        except Exception as e:
            print(f"Error fetching batch {batch}: {e}")
            continue

    # Step 4: Write name_rank file
    with open(name_rank_file, 'w') as f:
        for tid in taxids:
            name, rank = tax_info.get(tid, ("NA", "NA"))
            f.write(f"{tid}\t{name}\t{rank}\n")

    # Step 5: Join with input counts
    with open(joined_output_file, 'w') as out:
        with open(input_file, 'r') as f_in:
            for line in f_in:
                fields = line.strip().split('\t')
                if len(fields) < 2:
                    continue
                tid = fields[1]
                name, rank = tax_info.get(tid, ("NA", "NA"))
                out.write(f"{fields[0]}\t{tid}\t{name}\t{rank}\n")

def reformat_extra_taxids(input_file, output_reads_taxon, output_more_taxids):
    # Step 1: Open files
    with open(input_file, 'r') as infile, \
         open(output_reads_taxon, 'w') as out_clean, \
         open(output_more_taxids, 'w') as out_extra:

        for line in infile:
            fields = line.rstrip('\n').split('\t')

            if len(fields) > 5:
                # Write full original line to extra_TaxIDs_retrieved.tsv
                out_extra.write(line)
                # Write selected columns to ReadsTaxon_NamesRanks.tsv
                # Fields: count, taxID, name, rank, extra
                try:
                    out_clean.write(f"{fields[0]}\t{fields[1]}\t{fields[3]}\t{fields[4]}\t{fields[5]}\n")
                except IndexError:
                    # If some fields are missing, skip or log
                    continue
            else:
                # Write the whole line as-is to cleaned output
                out_clean.write(line)

    # Step 2: Ensure the "extra" file exists even if unused
    if os.path.getsize(output_more_taxids) == 0:
        # Leave it empty but ensure file exists
        open(output_more_taxids, 'a').close()

def map_falcon_accessions_to_taxids(input_file, output_file, email, api_key=None, verbose=False):
    """
    Reads a FALCON output file with <similarity>\t<accession>, replaces accessions with taxIDs,
    and writes <similarity>\t<taxID> to the output.

    Parameters:
        input_file (str): Path to the FALCON input file.
        output_file (str): Path to write the taxID-mapped output.
        email (str): Your email address for Entrez.
        api_key (str, optional): NCBI API key.
        verbose (bool): If True, print warnings.
    """
    from collections import OrderedDict

    # Step 1: Read accession and similarity into an ordered map
    accession_map = OrderedDict()
    with open(input_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue
            similarity, accession = fields
            accession_map[accession] = float(similarity)

    # Convert accession_map to a list of accessions
    accessions = [str(acc) for acc in accession_map.keys()]

    # Step 2: Resolve accession to taxID
    taxid_map = accession_to_taxid(
        accessions,
        email=email,
        api_key=api_key,
        verbose=verbose
    )

    # Step 3: Write new file with <similarity>\t<taxID>
    with open(output_file, 'w') as out:
        for acc, similarity in accession_map.items():
            taxid = taxid_map.get(acc, "0")
            out.write(f"{similarity:.3f}\t{taxid}\n")


    print(f"[✓] Written: {output_file}")

def accession_to_taxid(accessions, email, api_key=None, sleep_time=0.34, verbose=False):
    """
    Query NCBI Entrez to map accession numbers to taxonomic IDs (taxIDs).

    Parameters:
        accessions (list[str]): List of GenBank/RefSeq accession numbers (e.g., 'NC_001925.1').
        email (str): Your email address (required by NCBI Entrez).
        api_key (str, optional): NCBI API key for higher rate limits.
        sleep_time (float): Delay between batches to respect NCBI limits.
        verbose (bool): Whether to print warnings and debugging info.

    Returns:
        dict[str, str]: Mapping of accession -> taxID.
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    taxid_map = {}
    batch_size = 200

    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        if verbose:
            print(f"\n[📦] Querying batch {i // batch_size + 1} ({len(batch)} accessions):")
            print("      " + ", ".join(batch))

        try:
            with Entrez.esummary(db="nucleotide", id=",".join(batch)) as handle:
                records = Entrez.read(handle)

            if verbose:
                print(f"[📄] {len(records)} records returned")

            for idx, record in enumerate(records):
                acc = record.get("AccessionVersion")
                taxid = record.get("TaxId")

                if verbose:
                    print(f"\n → Record {idx + 1}:")
                    print(f"    AccessionVersion: {acc}")
                    print(f"    TaxId: {taxid}")

                if acc and taxid is not None:
                    try:
                        taxid_map[acc] = str(int(taxid))
                        if verbose:
                            print(f"    ✅ Parsed TaxId: {taxid_map[acc]}")
                    except (ValueError, TypeError) as parse_err:
                        if verbose:
                            print(f"    ❌ Could not parse TaxId for {acc}: {taxid} ({type(taxid)})")
                else:
                    if verbose:
                        print(f"    ⚠️  Missing Accession or TaxId. Accession: {acc}, TaxId: {taxid}")

            time.sleep(sleep_time)

        except Exception as e:
            print(f"\n[❌] Failed to fetch batch {i // batch_size + 1}:")
            print(f"    Accessions: {batch}")
            print(f"    Error: {e}")

    if verbose:
        print("\n[✅] Completed accession to TaxID mapping.")
        print(f"     {len(taxid_map)} of {len(accessions)} accessions mapped.\n")

    return taxid_map

def merge_readsp_taxon_tables(classifier_output_dir, output_file):
    """
    Merge per-sample <sample>_ReadspTaxon.txt files from a classifier output directory
    into a single count table (TSV) with taxIDs as rows and samples as columns.

    Parameters:
        classifier_output_dir (str): Directory like 'Centrifuge_output/'.
        output_file (str): Path to final merged table, e.g. 'Centrifuge_output/count_table.tsv'.
    """
    all_files = glob(os.path.join(classifier_output_dir, "*", "*_ReadspTaxon.txt"))
    data_frames = []
    sample_names = []

    for file_path in sorted(all_files):
        sample = os.path.basename(file_path).replace("_ReadspTaxon.txt", "")
        sample_names.append(sample)

        if os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path, sep="\t", header=None, names=["count", "taxID"])
            df = df.groupby("taxID", as_index=False).sum()
            df.rename(columns={"count": sample}, inplace=True)
        else:
            # Empty file → placeholder row for taxID = 0
            df = pd.DataFrame({"taxID": [0], sample: [0]})

        data_frames.append(df)

    if not data_frames:
        # No data to merge
        pd.DataFrame(columns=["taxID"]).to_csv(output_file, sep="\t", index=False)
        return

    # Merge all dataframes on 'taxID'
    merged_df = data_frames[0]
    for df in data_frames[1:]:
        merged_df = pd.merge(merged_df, df, on="taxID", how="outer")

    # Replace NA with 0 and ensure integer counts
    merged_df.fillna(0, inplace=True)
    merged_df[sample_names] = merged_df[sample_names].astype(int)

    # Write to file
    merged_df.to_csv(output_file, sep="\t", index=False)

def extract_accessions_from_fna(folder_path):
    """
    Extract accession numbers from the first line of each .fna file.

    Returns:
        list[str]: Accession numbers.
    """
    accessions = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".fna"):
            with open(os.path.join(folder_path, filename), 'r') as f:
                for line in f:
                    if line.startswith(">"):
                        acc = line.split()[0][1:]  # removes '>' and takes accession
                        accessions.append(acc)
                        break
    return accessions

def generate_ground_truth_from_fna_folder(folder_path, email, api_key=None, output_file="ground_truth.txt", verbose=False):
    """
    Extract accessions from .fna files in folder, map them to taxIDs using Entrez, and write taxIDs to output file.

    Parameters:
        folder_path (str): Path to the folder with .fna files.
        email (str): Your email address for NCBI Entrez.
        api_key (str, optional): Optional NCBI API key.
        output_file (str): Path to save the ground truth taxid list.
        verbose (bool): Print debug info if True.
    """
    accessions = extract_accessions_from_fna(folder_path)
    taxid_map = accession_to_taxid(accessions, email=email, api_key=api_key, verbose=verbose)

    with open(output_file, 'w') as f:
        for taxid in sorted(set(taxid_map.values())):
            f.write(f"{taxid}\n")

    if verbose:
        print(f"[✅] Ground truth saved to {output_file}")

def read_csv(file_path):
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    Each dictionary represents a row in the CSV file with column headers as keys.
    """
    import csv
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def detect_tool(input_path):
    """
    Detects classification tool name based on filename or path using keyword matching.

    Returns:
        str: Tool name ('kraken', 'centrifuge', 'kaiju', etc.), or 'unknown'
    """
    tool_patterns = {
        "kraken": r"kraken",
        "centrifuge": r"centrifuge",
        "kaiju": r"kaiju",
        "diamond": r"diamond",
        "metaphlan": r"metaphlan",
        "falcon": r"falcon",
        "blast": r"blast",
        "megablast": r"megablast",
        "clark": r"clark",
    }

    lower_path = input_path.lower()
    for tool, pattern in tool_patterns.items():
        if re.search(pattern, lower_path):
            return tool

    return "unknown"

def process_sample(input_path, sample, output_dir, email, api_key):
    sample_dir = os.path.join(output_dir, sample)
    os.makedirs(sample_dir, exist_ok=True)
    is_falcon = False

    # File paths
    raw_counts = os.path.join(sample_dir, f"{sample}_ReadspTaxon.txt")
    counts_with_taxid = os.path.join(sample_dir, f"{sample}_ReadspTaxon.txt")
    names_ranks = os.path.join(sample_dir, f"{sample}_NamesRanks.tsv")
    joined_table = os.path.join(sample_dir, f"{sample}_joined.tsv")
    empty_namesranks = os.path.join(sample_dir, f"{sample}_ReadsTaxon_NamesRanks.tsv")
    empty_correct_incorrect = os.path.join(sample_dir, f"{sample}_Correct_Incorrect.tsv")

    # Step 1
    # Detect type of tool
    if detect_tool(input_path=input_path) == "centrifuge":
        print(f"[🦠] Processing {sample} with Centrifuge")
        reads_per_taxon_centrifuge(
            input_file=input_path,
            output_file=raw_counts,
            empty_namesranks=empty_namesranks,
            empty_correct_incorrect=empty_correct_incorrect
        )
    elif detect_tool(input_path=input_path) == "kraken":
        print(f"[🦠] Processing {sample} with Kraken2")
        reads_per_taxon_kraken2(
            input_file=input_path,
            output_file=raw_counts,
            empty_namesranks=empty_namesranks,
            empty_correct_incorrect=empty_correct_incorrect
        )
    elif detect_tool(input_path=input_path) == "diamond":
        print(f"[🦠] Processing {sample} with DIAMOND")
        reads_per_taxon_diamond(
            input_file=input_path,
            output_file=raw_counts,
            empty_namesranks=empty_namesranks,
            empty_correct_incorrect=empty_correct_incorrect
        )
    elif detect_tool(input_path=input_path) == "metaphlan":
        print(f"[🦠] Processing {sample} with MetaPhlAn2")
        reads_per_taxon_metaphlan2(
            input_file=input_path,
            output_counts=raw_counts,
            output_names=os.path.join(sample_dir, f"{sample}_ReadspTaxon_Names.txt"),
            empty_namesranks=empty_namesranks,
            empty_correct_incorrect=empty_correct_incorrect
        )
    elif detect_tool(input_path=input_path) == "clark":
        print(f"[🦠] Processing {sample} with CLARK")
        reads_per_taxon_clark(
            input_file=input_path,
            output_file=raw_counts,
            empty_namesranks=empty_namesranks,
            empty_correct_incorrect=empty_correct_incorrect
        )
    elif detect_tool(input_path=input_path) == "kaiju":
        print(f"[🦠] Processing {sample} with Kaiju")
        reads_per_taxon_kaiju(
            input_file=input_path,
            output_file=raw_counts,
            empty_namesranks=empty_namesranks,
            empty_correct_incorrect=empty_correct_incorrect
        )
    elif detect_tool(input_path=input_path) == "falcon":
        print(f"[🧬] Processing {sample} with FALCON")
        is_falcon = True
        reads_per_taxon_falcon(
            input_file=input_path,
            output_file=raw_counts,
            empty_namesranks=empty_namesranks,
            empty_correct_incorrect=empty_correct_incorrect
        )
    else:
        print(f"[❓] Unknown tool detected for {sample}. Please check the input file.")
        # If unknown, just copy the input to raw_counts
        if not os.path.exists(raw_counts):
            with open(input_path, 'r') as src, open(raw_counts, 'w') as dst:
                dst.write(src.read())

    # Step 2: If FALCON, map accession to taxID
    final_input = raw_counts

    if is_falcon:
        print(f"[🧬] Mapping accessions to taxIDs for {sample} (FALCON detected)")
        map_falcon_accessions_to_taxids(
            input_file=raw_counts,
            output_file=counts_with_taxid,
            email=email,
            api_key=api_key
        )
        final_input = counts_with_taxid

    # Step 3
    get_name_rank(
        input_file=final_input,
        name_rank_file=names_ranks,
        joined_output_file=joined_table,
        email=email
    )

def main():
    parser = argparse.ArgumentParser(description="Taxonomy processing pipeline")
    parser.add_argument("--input", required=True, help="Report file or folder of files")
    parser.add_argument("--output_dir", default="output", help="Where to write outputs")
    parser.add_argument("--email", required=True, help="NCBI Entrez email")
    parser.add_argument("--api_key", default=None, help="NCBI API key")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine files to process
    if os.path.isdir(args.input):
        input_files = glob(os.path.join(args.input, "*.txt"))
    elif os.path.isfile(args.input):
        input_files = [args.input]
    else:
        raise FileNotFoundError(f"Input path '{args.input}' is not valid.")

    # Process each input file
    for file_path in input_files:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        match = re.search(r"depth\d+_read\d+_deam[\d\.]+", basename)
        sample_name = match.group(0) if match else basename
        print(f"\n[▶] Processing sample: {sample_name}")
        process_sample(
            input_path=file_path,
            sample=sample_name,
            output_dir=args.output_dir,
            email=args.email,
            api_key=args.api_key
        )

    # Step 4: Merge all counts across samples
    print("\n[📊] Merging ReadspTaxon files into count table...")
    merge_readsp_taxon_tables(
        classifier_output_dir=args.output_dir,
        output_file=os.path.join(args.output_dir, "count_table.tsv")
    )
    print("[✅] Done.")

if __name__ == "__main__":
    main()

    # accessions = ['GCF_000839645.1', 'GCF_000846365.1', 'GCF_000859885.1', 'GCF_000859985.2', 'GCF_000861825.2', 'GCF_000863805.1', 'NC_012920.1', 'GCF_000005845.2']
    #
    # for accession in accessions:
    #     print(f"Processing accession: {accession}")
    #     taxid_map = [accession]
    #     accession_to_taxid(accessions=taxid_map, email="luiscarloslaranjeira@hotmail.com")

    # generate_ground_truth_from_fna_folder(
    #     folder_path="D:/Data/data/bact",
    #     email="your.email@example.com",
    #     verbose=True
    # )

    # reads_per_taxon_falcon(
    #     input_file="D:\Data\\reports_output\FALCON2\sim_depth1_read40_deam0_s.fq_report.txt",
    #     output_file="sample123_ReadspTaxon.txt",
    #     empty_namesranks="sample123_ReadsTaxon_NamesRanks.tsv",
    #     empty_correct_incorrect="sample123_Correct_Incorrect.tsv"
    # )
    #
    # map_falcon_accessions_to_taxids(
    #     input_file="sample123_ReadspTaxon.txt",
    #     output_file="sample123_ReadspTaxon_tx.txt",
    #     email="your.email@example.com",
    # )
    #
    # get_name_rank(
    #     input_file="sample123_ReadspTaxon_tx.txt",
    #     name_rank_file="sample1_NamesRanks.tsv",
    #     joined_output_file="sample1_joined.tsv",
    #     email="your.email@example.com",
    # )
    #
    # merge_readsp_taxon_tables("Centrifuge_output", "Centrifuge_output/count_table.tsv")


import argparse
import re

def parse_float_line(line):
    """
    Parses a line starting with '[000' and extracts the index and float values.
    Returns a tuple (index, list_of_floats) or None if the line doesn't match.
    """
    match = re.match(r'^\[(\d+)\]:\s*(.*)', line)
    if match:
        index = int(match.group(1))
        # Remove the non-breaking space (U+00A0) often found at the end of these lines
        float_str_list = match.group(2).replace('\xa0', '').strip().split()
        try:
            float_values = [float(f) for f in float_str_list]
            return index, float_values
        except ValueError:
            # Handle cases where some "float" values might not be valid numbers
            return None
    return None

def compare_files(file1_path, file2_path):
    """
    Compares the float data lines in two files.
    """
    data1 = {}
    data2 = {}
    
    print(f"Processing '{file1_path}'...")
    with open(file1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            parsed = parse_float_line(line)
            if parsed:
                index, values = parsed
                data1[index] = values

    print(f"Processing '{file2_path}'...")
    with open(file2_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            parsed = parse_float_line(line)
            if parsed:
                index, values = parsed
                data2[index] = values

    print("\n--- Comparison Results ---")

    all_indices = sorted(set(data1.keys()) | set(data2.keys()))

    differences_found = False
    for index in all_indices:
        values1 = data1.get(index)
        values2 = data2.get(index)

        if values1 is None and values2 is None:
            continue # Should not happen with current logic if all_indices is built correctly

        if values1 is None:
            print(f"Index [{index:06d}]: Missing in '{file1_path}'")
            differences_found = True
        elif values2 is None:
            print(f"Index [{index:06d}]: Missing in '{file2_path}'")
            differences_found = True
        else:
            if len(values1) != len(values2):
                print(f"Index [{index:06d}]: Different number of elements.")
                print(f"  File 1 ({file1_path}): {len(values1)} elements")
                print(f"  File 2 ({file2_path}): {len(values2)} elements")
                differences_found = True
            else:
                # Compare float values with a small tolerance for floating point inaccuracies
                # You can adjust this tolerance (e.g., 1e-6, 1e-9) based on your needs.
                tolerance = 1e-5
                
                mismatched_elements = []
                for i, (val1, val2) in enumerate(zip(values1, values2)):
                    if abs(val1 - val2) > tolerance:
                        mismatched_elements.append((i, val1, val2))
                
                if mismatched_elements:
                    print(f"Index [{index:06d}]: Mismatched float values found!")
                    for pos, v1, v2 in mismatched_elements:
                        print(f"  Element {pos}: File 1 = {v1:.3f}, File 2 = {v2:.3f}")
                    differences_found = True
                # else:
                #     print(f"Index [{index:06d}]: Values are identical.") # Uncomment for detailed success messages

    if not differences_found:
        print("\nAll comparable float lines are identical across both files.")
    else:
        print("\nDifferences were found between the files.")

def main():
    parser = argparse.ArgumentParser(description="Compare float data lines from two log files.")
    parser.add_argument("file1", help="Path to the first log file.")
    parser.add_argument("file2", help="Path to the second log file.")
    
    args = parser.parse_args()
    
    compare_files(args.file1, args.file2)

if __name__ == "__main__":
    main()
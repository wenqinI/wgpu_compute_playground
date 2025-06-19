
import sys

def extract_between_to_list(filename, start_marker, end_marker):
    result = []
    with open(filename, 'r') as f:
        inside = False
        for line in f:
            if start_marker in line:
                inside = True
                continue
            if end_marker in line:
                break
            if inside:
                result.append(line.rstrip('\n'))
    return result

def diff_files(file1, file2, start_marker, end_marker):
    lines1 = extract_between_to_list(file1, start_marker, end_marker)
    lines2 = extract_between_to_list(file2, start_marker, end_marker)

    max_len = max(len(lines1), len(lines2))
    for i in range(max_len):
        line1 = lines1[i].rstrip('\n') if i < len(lines1) else ''
        line2 = lines2[i].rstrip('\n') if i < len(lines2) else ''
        if line1 != line2:
            print(f"- {line1}")
            print(f"+ {line2}")
            return True
    return  False


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python diff.py file1 file2 start_marker end_marker")
        sys.exit(1)
    sys.exit(diff_files(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))

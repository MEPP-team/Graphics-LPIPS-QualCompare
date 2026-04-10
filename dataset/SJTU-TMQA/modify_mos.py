# modify_mos.py
import csv
import sys

def modify_mos(input_csv, output_csv):
    # Open input and output CSV files
    with open(input_csv, "r", newline="", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # Read header
        header = next(reader)

        # Find index of "name" column
        try:
            mos_idx = header.index("MOS")
            cimin = header.index("CI_MIN")
            cimax = header.index("CI_MAX")
        except ValueError:
            raise ValueError('Column "MOS" not found in header.')

        # Process rows
        for row in reader:
            # Get MOS
            mos_value = row[mos_idx]
            ci_min_value = row[cimin]
            ci_max_value = row[cimax]
            # This MOS value is between 0 and 10, 0 being the worst, 10 the best.
            # We will change it so the new range is 1 to 5, 1 being the worst.
            # Formula to convert: new_mos = mos_value * 0.4 + 1
            try:
                mos_float = float(mos_value)
                new_mos = mos_float * 0.4 + 1
                new_ci_min = float(ci_min_value) * 0.4 + 1
                new_ci_max = float(ci_max_value) * 0.4 + 1
                new_ci_min_str = f"{new_ci_min:.2f}"
                new_ci_max_str = f"{new_ci_max:.2f}"
                new_mos_str = f"{new_mos:.2f}"
            except ValueError:
                new_mos_str = mos_value  # In case of conversion error, keep original            
                new_ci_min = ci_min_value
                new_ci_max = ci_max_value
            # replace MOS value with new value
            row[mos_idx] = new_mos_str
            row[cimin] = new_ci_min_str
            row[cimax] = new_ci_max_str
            # Write modified row to output CSV
            writer.writerow(row)
            
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_MOS.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    modify_mos(input_csv, output_csv)

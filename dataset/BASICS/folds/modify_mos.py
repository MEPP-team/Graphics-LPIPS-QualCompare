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
            mos_idx = header.index("mos")
            ci = header.index("ci")
        except ValueError:
            raise ValueError('Column "MOS" not found in header.')

        # Process rows
        for row in reader:
            # Get MOS
            mos_value = row[mos_idx]
            ci_value = row[ci]
            # This MOS value is between 1 and 5, 1 being the worst, 5 the best.
            # We will change it so the new range 0 to 1, 1 being the worst.
            # Formula to convert: new_mos = (5 - mos_value) * 0.25
            try:
                new_mos = (5 - float(mos_value)) * 0.25
                new_ci = (5 - float(ci_value)) * 0.25
                new_ci_str = f"{new_ci:.2f}"
                new_mos_str = f"{new_mos:.2f}"
            except ValueError:
                new_mos_str = mos_value  # In case of conversion error, keep original            
                new_ci_str = ci_value
            # replace MOS value with new value
            row[mos_idx] = new_mos_str
            row[ci] = new_ci_str
            # Write modified row to output CSV
            writer.writerow(row)
            
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_MOS.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    modify_mos(input_csv, output_csv)

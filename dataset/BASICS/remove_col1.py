# add_model_column.py
import csv
import sys

def remove_column_1(input_csv, output_csv):
    # Open input and output CSV files
    with open(input_csv, "r", newline="", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # Read header
        header = next(reader)

        new_header = header[1:]
        writer.writerow(new_header)

        # Process rows
        for row in reader:
            # Get object name
            new_row = row[1:]
            writer.writerow(new_row)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_col1.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    remove_column_1(input_csv, output_csv)



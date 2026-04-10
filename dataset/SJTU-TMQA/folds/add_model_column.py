# add_model_column.py
import csv
import sys

def add_model_column(input_csv, output_csv):
    # Open input and output CSV files
    with open(input_csv, "r", newline="", encoding="utf-8") as fin, \
         open(output_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # Read header
        header = next(reader)

        # Find index of "name" column
        try:
            name_idx = header.index("name")
        except ValueError:
            raise ValueError('Column "name" not found in header.')

        # New header with "model" as first column
        new_header = ["model"] + header
        writer.writerow(new_header)

        # Process rows
        for row in reader:
            # Get object name
            name_value = row[name_idx]

            # Take part before first underscore, or full string if no underscore
            model = name_value.split("_", 1)[0]

            # Write new row with model as first column
            new_row = [model] + row
            writer.writerow(new_row)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_model_column.py input.csv output.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    add_model_column(input_csv, output_csv)

from collections.abc import Iterable

import numpy as np
import pandas as pd

def process_frame(df: pd.DataFrame, animal_name: str):
    # Group the rows based on their frequency value.
    grouped_dfs = df.groupby(df['f2(Hz)'])

    def gen_data():
        # Process each frequency, find the minimum dB that is within 3 std of the '2f1-f2Nse(dB)' column.
        for _, freq_i in grouped_dfs:
            # Get the current frequency in kHz
            freq = freq_i["f2(Hz)"].iloc[0] / 1000

            # Compute the sound-noise ratio for this frequency
            sound_noise_ratio = freq_i['2f1-f2Nse(dB)'] + 6

            # Filter out values that under the sound noise ratio
            filtered = freq_i[freq_i['2f1-f2(dB)'] >= sound_noise_ratio]

            # # Find the minimum dB of the filtered values
            min_dB = float(filtered.min()[':dB'])
            min_dB = 80 if np.isnan(min_dB) else min_dB

            yield freq, min_dB, animal_name

    output = pd.DataFrame(gen_data(), columns=["f(kHz)", "value", "animal"])
    return output.pivot(index="f(kHz)", columns="animal", values="value")

def process(entries: Iterable[tuple[pd.DataFrame, str]]):
    result_df = pd.DataFrame()

    for df, animal_name in entries:
        output_df = process_frame(df, animal_name)
        result_df = pd.concat([ result_df, output_df ], axis=1)

    return result_df

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from datetime import datetime
    from pathlib import Path
    import os
    import sys

    # Parse cmdline arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    _ = parser.add_argument('-p', dest="pattern", type=str, default='DP-1.TSV', help='Pattern to match for input file selection')
    _ = parser.add_argument('-o', dest="output", type=Path, default='DP_Processed_Data', help='Name of the output directory')
    _ = parser.add_argument('-i', dest="inputs", nargs='+', type=Path, required=True, default='.', help='List of input directories containing files to process')

    args = parser.parse_args()
    pattern: str = args.pattern
    output: Path = args.output
    inputs: Iterable[Path] = args.inputs

    # Construct output path
    output_path = output.absolute()
    os.makedirs(output_path, exist_ok=True)

    def create_dataframes(search_path: Path):
        def find_data_start_index(file_path: Path):
            with open(file_path, 'r') as file:
                for i, line in enumerate(file):
                    if line.startswith(':DATA'):
                        return i

        for file_path in filter(Path.is_file, search_path.rglob(pattern)):
            print(f'Processing \'{file_path}\'...')

            # Find start of the data
            data_start = find_data_start_index(file_path)
            if data_start is None:
                print('Could not find data due to unexpected formatting, skipping...', file=sys.stderr)
                continue

            # Read in the file as a dataframe, remove any rows that contain NA values
            df = pd.read_csv(file_path, sep='\t', skiprows=data_start-1)
            df.dropna(axis=0, how='any', inplace=True)
            yield df, file_path.parent.name

    # Create an output for each input
    for search_path in inputs:
        output_df = process(create_dataframes(search_path))
        output_name = f'{search_path.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
        output_file = os.path.join(output_path, output_name)

        with open(output_file, 'w') as f:
            output_df.to_csv(f, lineterminator='\n')
            print(f'Output written to \'{output_file}\'')

if __name__ == "__main__":
    main()

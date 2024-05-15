import re

def is_file_format_correct(file):
    """Check if the file format is correct."""
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not re.match(r'Seq_\d{3}_\d+/imgs/.*\.png', line.strip()):
                return False
    return True

def is_sample_list_length_valid(file):
    """Check if the sample list length is no more than 400."""
    with open(file, 'r') as f:
        lines = f.readlines()
        if len(lines) > 400:
            return False
    return True


# Check whether the next line in sample content is repeated
def check_repeated_samples(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        if len(lines) != len(set(lines)):
            return False
    return True


def main():
    """Main function to run the checks."""
    file_path = 'data/output/samples_xtest.txt'
    
    if is_file_format_correct(file_path):
        print('Congrats! File format is correct.')
    else:
        print('File format is incorrect, please check!')

    if is_sample_list_length_valid(file_path):
        print('Congrats! Sample list length is less than 400.')
    else:  
        print('Warning: Sample list length is more than 400, only the first 400 samples will be used for evaluation')

    if not check_repeated_samples(file_path):
        print('Your sample list contains repeated samples, please check!')

if __name__ == "__main__":
    main()
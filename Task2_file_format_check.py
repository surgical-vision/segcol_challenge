import re

# Check if the file format is correct
def check_file_format(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not re.match(r'Seq_\d{3}_\d+/imgs/.*\.png', line.strip()): #/frame_\d+\.png', line.strip()):
                return False
    return True

check_result = check_file_format('samples_xtest.txt')
if check_result:
    print('File format is correct')
else:
    print('File format is incorrect')
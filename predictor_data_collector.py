import os

folder_path = "meta_results"
output_file_path = "data.txt"

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the path is a file (not a subdirectory)
        if os.path.isfile(file_path):
            # Open each file in read mode and append its content to the output file
            with open(file_path, 'r') as input_file:
                file_content = input_file.read()
                output_file.write(file_content)
                # Add a newline to separate content from different files
                output_file.write('\n')

print("Data from all files in 'meta_results' has been successfully stored in 'data.txt'.")

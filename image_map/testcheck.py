import pandas as pd

# Function to check for self-pair and duplicate pairs
def check_image_pairs(file_path):
    df = pd.read_csv(file_path)
    
    # Check for self-pairs
    self_pairs = df[df['image_path1'] == df['image_path2']]
    if not self_pairs.empty:
        print(f"Self-pairs found in {file_path}:")
        print(self_pairs)
    
    # Check for duplicate pairs (ignoring order)
    pairs_set = set()
    duplicates = []
    for _, row in df.iterrows():
        pair = tuple(sorted([row['image_path1'], row['image_path2']]))
        if pair in pairs_set:
            duplicates.append(pair)
        else:
            pairs_set.add(pair)
    
    if duplicates:
        print(f"Duplicate pairs found in {file_path}:")
        for pair in duplicates:
            print(pair)
    else:
        print(f"No duplicate pairs found in {file_path}.")

# Paths to the CSV files
train_file_path = 'train_image_pairs.csv'
test_file_path = 'test_image_pairs.csv'

# Check the train and test files
check_image_pairs(train_file_path)
check_image_pairs(test_file_path)



# Function to find PNG files that appear in both CSV files
def find_common_pngs(train_file, test_file):
    # Read the CSV files
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Get the set of all PNG files in each CSV file
    train_pngs = set(train_df['image_path1']).union(set(train_df['image_path2']))
    test_pngs = set(test_df['image_path1']).union(set(test_df['image_path2']))
    
    # Find common PNG files
    common_pngs = train_pngs.intersection(test_pngs)
    
    if common_pngs:
        print("PNG files found in both CSV files:")
        for png in common_pngs:
            print(png)
    else:
        print("No common PNG files found between the two CSV files.")
        
# Check for common PNG files
find_common_pngs(train_file_path, test_file_path)
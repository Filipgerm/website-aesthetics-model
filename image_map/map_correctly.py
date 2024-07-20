import pandas as pd

# Data folder paths
data_folder = '../../Calista/website-aesthetics-datasets-master/rating-based-dataset/preprocess/'
train_data_path = data_folder + 'train_means_list.csv'
test_data_path = data_folder + 'test_list.csv'
comparison_dataset = '../../Calista/website-aesthetics-datasets-master/comparison-based-dataset/'
comparison_images_path = comparison_dataset + 'resized_images'
comparison_data_path = comparison_dataset + 'data/comparisons.csv'

# Step 1: Read input files
train_means_list = pd.read_csv(train_data_path)
test_list = pd.read_csv(test_data_path)
comparisons = pd.read_csv(comparison_data_path)

# Step 2: Create the mapping dictionary from the provided list
mapping = {
    '0.png': '/english_resized/0.png',
    '1.png': '/english_resized/1.png',
    '2.png': '/english_resized/10.png',
    '3.png': '/english_resized/100.png',
    '4.png': '/english_resized/101.png',
    '5.png': '/english_resized/102.png',
    '6.png': '/english_resized/103.png',
    '7.png': '/english_resized/104.png',
    '8.png': '/english_resized/105.png',
    '9.png': '/english_resized/106.png',
    '10.png': '/english_resized/107.png',
    '11.png': '/english_resized/108.png',
    '12.png': '/english_resized/109.png',
    '13.png': '/english_resized/11.png',
    '14.png': '/english_resized/111.png',
    '15.png': '/english_resized/112.png',
    '16.png': '/english_resized/113.png',
    '17.png': '/english_resized/114.png',
    '18.png': '/english_resized/115.png',
    '19.png': '/english_resized/116.png',
    '20.png': '/english_resized/117.png',
    '21.png': '/english_resized/118.png',
    '22.png': '/english_resized/119.png',
    '23.png': '/english_resized/12.png',
    '24.png': '/english_resized/120.png',
    '25.png': '/english_resized/121.png',
    '26.png': '/english_resized/122.png',
    '27.png': '/english_resized/123.png',
    '28.png': '/english_resized/124.png',
    '29.png': '/english_resized/125.png',
    '30.png': '/english_resized/128.png',
    '31.png': '/english_resized/129.png',
    '32.png': '/english_resized/13.png',
    '33.png': '/english_resized/130.png',
    '34.png': '/english_resized/131.png',
    '35.png': '/english_resized/132.png',
    '36.png': '/english_resized/133.png',
    '37.png': '/english_resized/134.png',
    '38.png': '/english_resized/135.png',
    '39.png': '/english_resized/136.png',
    '40.png': '/english_resized/137.png',
    '41.png': '/english_resized/138.png',
    '42.png': '/english_resized/139.png',
    '43.png': '/english_resized/14.png',
    '44.png': '/english_resized/140.png',
    '45.png': '/english_resized/141.png',
    '46.png': '/english_resized/142.png',
    '47.png': '/english_resized/143.png',
    '48.png': '/english_resized/144.png',
    '49.png': '/english_resized/145.png',
    '50.png': '/english_resized/146.png',
    '51.png': '/english_resized/147.png',
    '52.png': '/english_resized/148.png',
    '53.png': '/english_resized/149.png',
    '54.png': '/english_resized/15.png',
    '55.png': '/english_resized/150.png',
    '56.png': '/english_resized/151.png',
    '57.png': '/english_resized/152.png',
    '58.png': '/english_resized/153.png',
    '59.png': '/english_resized/154.png',
    '60.png': '/english_resized/155.png',
    '61.png': '/english_resized/156.png',
    '62.png': '/english_resized/157.png',
    '63.png': '/english_resized/158.png',
    '64.png': '/english_resized/159.png',
    '65.png': '/english_resized/16.png',
    '66.png': '/english_resized/160.png',
    '67.png': '/english_resized/161.png',
    '68.png': '/english_resized/162.png',
    '69.png': '/english_resized/163.png',
    '70.png': '/english_resized/164.png',
    '71.png': '/english_resized/165.png',
    '72.png': '/english_resized/166.png',
    '73.png': '/english_resized/167.png',
    '74.png': '/english_resized/168.png',
    '75.png': '/english_resized/169.png',
    '76.png': '/english_resized/17.png',
    '77.png': '/english_resized/170.png',
    '78.png': '/english_resized/171.png',
    '79.png': '/english_resized/172.png',
    '80.png': '/english_resized/173.png',
    '81.png': '/english_resized/174.png',
    '82.png': '/english_resized/175.png',
    '83.png': '/english_resized/176.png',
    '84.png': '/english_resized/177.png',
    '85.png': '/english_resized/178.png',
    '86.png': '/english_resized/179.png',
    '87.png': '/english_resized/18.png',
    '88.png': '/english_resized/180.png',
    '89.png': '/english_resized/181.png',
    '90.png': '/english_resized/182.png',
    '91.png': '/english_resized/183.png',
    '92.png': '/english_resized/184.png',
    '93.png': '/english_resized/185.png',
    '94.png': '/english_resized/186.png',
    '95.png': '/english_resized/187.png',
    '96.png': '/english_resized/188.png',
    '97.png': '/english_resized/189.png',
    '98.png': '/english_resized/19.png',
    '99.png': '/english_resized/190.png',
    '100.png': '/english_resized/191.png',
    '101.png': '/english_resized/192.png',
    '102.png': '/english_resized/193.png',
    '103.png': '/english_resized/194.png',
    '104.png': '/english_resized/195.png',
    '105.png': '/english_resized/196.png',
    '106.png': '/english_resized/197.png',
    '107.png': '/english_resized/198.png',
    '108.png': '/english_resized/199.png',
    '109.png': '/english_resized/2.png',
    '110.png': '/english_resized/20.png',
    '111.png': '/english_resized/200.png',
    '112.png': '/english_resized/201.png',
    '113.png': '/english_resized/202.png',
    '114.png': '/english_resized/203.png',
    '115.png': '/english_resized/204.png',
    '116.png': '/english_resized/205.png',
    '117.png': '/english_resized/206.png',
    '118.png': '/english_resized/207.png',
    '119.png': '/english_resized/208.png',
    '120.png': '/english_resized/209.png',
    '121.png': '/english_resized/21.png',
    '122.png': '/english_resized/210.png',
    '123.png': '/english_resized/211.png',
    '124.png': '/english_resized/212.png',
    '125.png': '/english_resized/213.png',
    '126.png': '/english_resized/214.png',
    '127.png': '/english_resized/215.png',
    '128.png': '/english_resized/216.png',
    '129.png': '/english_resized/217.png',
    '130.png': '/english_resized/218.png',
    '131.png': '/english_resized/219.png',
    '132.png': '/english_resized/22.png',
    '133.png': '/english_resized/220.png',
    '134.png': '/english_resized/221.png',
    '135.png': '/english_resized/222.png',
    '136.png': '/english_resized/223.png',
    '137.png': '/english_resized/224.png',
    '138.png': '/english_resized/225.png',
    '139.png': '/english_resized/226.png',
    '140.png': '/english_resized/227.png',
    '141.png': '/english_resized/228.png',
    '142.png': '/english_resized/229.png',
    '143.png': '/english_resized/23.png',
    '144.png': '/english_resized/230.png',
    '145.png': '/english_resized/231.png',
    '146.png': '/english_resized/232.png',
    '147.png': '/english_resized/233.png',
    '148.png': '/english_resized/234.png',
    '149.png': '/english_resized/235.png',
    '150.png': '/english_resized/236.png',
    '151.png': '/english_resized/237.png',
    '152.png': '/english_resized/238.png',
    '153.png': '/english_resized/239.png',
    '154.png': '/english_resized/24.png',
    '155.png': '/english_resized/240.png',
    '156.png': '/english_resized/241.png',
    '157.png': '/english_resized/242.png',
    '158.png': '/english_resized/243.png',
    '159.png': '/english_resized/244.png',
    '160.png': '/english_resized/245.png',
    '161.png': '/english_resized/246.png',
    '162.png': '/english_resized/247.png',
    '163.png': '/english_resized/248.png',
    '164.png': '/english_resized/249.png',
    '165.png': '/english_resized/25.png',
    '166.png': '/english_resized/250.png',
    '167.png': '/english_resized/251.png',
    '168.png': '/english_resized/252.png',
    '169.png': '/english_resized/253.png',
    '170.png': '/english_resized/254.png',
    '171.png': '/english_resized/255.png',
    '172.png': '/english_resized/256.png',
    '173.png': '/english_resized/257.png',
    '174.png': '/english_resized/258.png',
    '175.png': '/english_resized/259.png',
    '176.png': '/english_resized/26.png',
    '177.png': '/english_resized/260.png',
    '178.png': '/english_resized/261.png',
    '179.png': '/english_resized/262.png',
    '180.png': '/english_resized/263.png',
    '181.png': '/english_resized/264.png',
    '182.png': '/english_resized/265.png',
    '183.png': '/english_resized/266.png',
    '184.png': '/english_resized/267.png',
    '185.png': '/english_resized/268.png',
    '186.png': '/english_resized/269.png',
    '187.png': '/english_resized/27.png',
    '188.png': '/english_resized/270.png',
    '189.png': '/english_resized/271.png',
    '190.png': '/english_resized/272.png',
    '191.png': '/english_resized/273.png',
    '192.png': '/english_resized/274.png',
    '193.png': '/english_resized/275.png',
    '194.png': '/english_resized/276.png',
    '195.png': '/english_resized/277.png',
    '196.png': '/english_resized/278.png',
    '197.png': '/english_resized/279.png',
    '198.png': '/english_resized/28.png',
    '199.png': '/english_resized/280.png',
    '200.png': '/english_resized/281.png',
    '201.png': '/english_resized/282.png',
    '202.png': '/english_resized/283.png',
    '203.png': '/english_resized/284.png',
    '204.png': '/english_resized/286.png',
    '205.png': '/english_resized/287.png',
    '206.png': '/english_resized/288.png',
    '207.png': '/english_resized/289.png',
    '208.png': '/english_resized/29.png',
    '209.png': '/english_resized/290.png',
    '210.png': '/english_resized/291.png',
    '211.png': '/english_resized/292.png',
    '212.png': '/english_resized/293.png',
    '213.png': '/english_resized/294.png',
    '214.png': '/english_resized/295.png',
    '215.png': '/english_resized/297.png',
    '216.png': '/english_resized/298.png',
    '217.png': '/english_resized/299.png',
    '218.png': '/english_resized/3.png',
    '219.png': '/english_resized/30.png',
    '220.png': '/english_resized/300.png',
    '221.png': '/english_resized/302.png',
    '222.png': '/english_resized/303.png',
    '223.png': '/english_resized/304.png',
    '224.png': '/english_resized/305.png',
    '225.png': '/english_resized/306.png',
    '226.png': '/english_resized/307.png',
    '227.png': '/english_resized/308.png',
    '228.png': '/english_resized/309.png',
    '229.png': '/english_resized/310.png',
    '230.png': '/english_resized/311.png',
    '231.png': '/english_resized/312.png',
    '232.png': '/english_resized/313.png',
    '233.png': '/english_resized/314.png',
    '234.png': '/english_resized/315.png',
    '235.png': '/english_resized/316.png',
    '236.png': '/english_resized/317.png',
    '237.png': '/english_resized/318.png',
    '238.png': '/english_resized/319.png',
    '239.png': '/english_resized/32.png',
    '240.png': '/english_resized/320.png',
    '241.png': '/english_resized/321.png',
    '242.png': '/english_resized/322.png',
    '243.png': '/english_resized/323.png',
    '244.png': '/english_resized/324.png',
    '245.png': '/english_resized/325.png',
    '246.png': '/english_resized/326.png',
    '247.png': '/english_resized/327.png',
    '248.png': '/english_resized/328.png',
    '249.png': '/english_resized/329.png',
    '250.png': '/english_resized/33.png',
    '251.png': '/english_resized/330.png',
    '252.png': '/english_resized/331.png',
    '253.png': '/english_resized/332.png',
    '254.png': '/english_resized/333.png',
    '255.png': '/english_resized/334.png',
    '256.png': '/english_resized/335.png',
    '257.png': '/english_resized/336.png',
    '258.png': '/english_resized/337.png',
    '259.png': '/english_resized/338.png',
    '260.png': '/english_resized/339.png',
    '261.png': '/english_resized/34.png',
    '262.png': '/english_resized/340.png',
    '263.png': '/english_resized/341.png',
    '264.png': '/english_resized/342.png',
    '265.png': '/english_resized/343.png',
    '266.png': '/english_resized/344.png',
    '267.png': '/english_resized/345.png',
    '268.png': '/english_resized/346.png',
    '269.png': '/english_resized/347.png',
    '270.png': '/english_resized/348.png',
    '271.png': '/english_resized/349.png',
    '272.png': '/english_resized/35.png',
    '273.png': '/english_resized/36.png',
    '274.png': '/english_resized/37.png',
    '275.png': '/english_resized/38.png',
    '276.png': '/english_resized/39.png',
    '277.png': '/english_resized/4.png',
    '278.png': '/english_resized/40.png',
    '279.png': '/english_resized/41.png',
    '280.png': '/english_resized/42.png',
    '281.png': '/english_resized/43.png',
    '282.png': '/english_resized/44.png',
    '283.png': '/english_resized/45.png',
    '284.png': '/english_resized/46.png',
    '285.png': '/english_resized/47.png',
    '286.png': '/english_resized/48.png',
    '287.png': '/english_resized/49.png',
    '288.png': '/english_resized/5.png',
    '289.png': '/english_resized/50.png',
    '290.png': '/english_resized/51.png',
    '291.png': '/english_resized/52.png',
    '292.png': '/english_resized/54.png',
    '293.png': '/english_resized/55.png',
    '294.png': '/english_resized/56.png',
    '295.png': '/english_resized/57.png',
    '296.png': '/english_resized/58.png',
    '297.png': '/english_resized/59.png',
    '298.png': '/english_resized/60.png',
    '299.png': '/english_resized/61.png',
    '300.png': '/english_resized/62.png',
    '301.png': '/english_resized/63.png',
    '302.png': '/english_resized/64.png',
    '303.png': '/english_resized/65.png',
    '304.png': '/english_resized/66.png',
    '305.png': '/english_resized/67.png',
    '306.png': '/english_resized/68.png',
    '307.png': '/english_resized/69.png',
    '308.png': '/english_resized/7.png',
    '309.png': '/english_resized/70.png',
    '310.png': '/english_resized/71.png',
    '311.png': '/english_resized/72.png',
    '312.png': '/english_resized/73.png',
    '313.png': '/english_resized/74.png',
    '314.png': '/english_resized/75.png',
    '315.png': '/english_resized/76.png',
    '316.png': '/english_resized/77.png',
    '317.png': '/english_resized/78.png',
    '318.png': '/english_resized/79.png',
    '319.png': '/english_resized/8.png',
    '320.png': '/english_resized/80.png',
    '321.png': '/english_resized/81.png',
    '322.png': '/english_resized/82.png',
    '323.png': '/english_resized/83.png',
    '324.png': '/english_resized/85.png',
    '325.png': '/english_resized/86.png',
    '326.png': '/english_resized/87.png',
    '327.png': '/english_resized/88.png',
    '328.png': '/english_resized/89.png',
    '329.png': '/english_resized/9.png',
    '330.png': '/english_resized/90.png',
    '331.png': '/english_resized/91.png',
    '332.png': '/english_resized/92.png',
    '333.png': '/english_resized/93.png',
    '334.png': '/english_resized/94.png',
    '335.png': '/english_resized/95.png',
    '336.png': '/english_resized/96.png',
    '337.png': '/english_resized/97.png',
    '338.png': '/english_resized/98.png',
    '339.png': '/english_resized/99.png',
    '340.png': '/foreign_resized/0.png',
    '341.png': '/foreign_resized/1.png',
    '342.png': '/foreign_resized/10.png',
    '343.png': '/foreign_resized/11.png',
    '344.png': '/foreign_resized/12.png',
    '345.png': '/foreign_resized/13.png',
    '346.png': '/foreign_resized/14.png',
    '347.png': '/foreign_resized/15.png',
    '348.png': '/foreign_resized/16.png',
    '349.png': '/foreign_resized/17.png',
    '350.png': '/foreign_resized/18.png',
    '351.png': '/foreign_resized/19.png',
    '352.png': '/foreign_resized/2.png',
    '353.png': '/foreign_resized/20.png',
    '354.png': '/foreign_resized/21.png',
    '355.png': '/foreign_resized/22.png',
    '356.png': '/foreign_resized/23.png',
    '357.png': '/foreign_resized/24.png',
    '358.png': '/foreign_resized/25.png',
    '359.png': '/foreign_resized/26.png',
    '360.png': '/foreign_resized/27.png',
    '361.png': '/foreign_resized/28.png',
    '362.png': '/foreign_resized/29.png',
    '363.png': '/foreign_resized/3.png',
    '364.png': '/foreign_resized/30.png',
    '365.png': '/foreign_resized/31.png',
    '366.png': '/foreign_resized/32.png',
    '367.png': '/foreign_resized/33.png',
    '368.png': '/foreign_resized/34.png',
    '369.png': '/foreign_resized/35.png',
    '370.png': '/foreign_resized/36.png',
    '371.png': '/foreign_resized/37.png',
    '372.png': '/foreign_resized/38.png',
    '373.png': '/foreign_resized/39.png',
    '374.png': '/foreign_resized/4.png',
    '375.png': '/foreign_resized/40.png',
    '376.png': '/foreign_resized/41.png',
    '377.png': '/foreign_resized/42.png',
    '378.png': '/foreign_resized/43.png',
    '379.png': '/foreign_resized/44.png',
    '380.png': '/foreign_resized/46.png',
    '381.png': '/foreign_resized/47.png',
    '382.png': '/foreign_resized/48.png',
    '383.png': '/foreign_resized/49.png',
    '384.png': '/foreign_resized/5.png',
    '385.png': '/foreign_resized/50.png',
    '386.png': '/foreign_resized/51.png',
    '387.png': '/foreign_resized/52.png',
    '388.png': '/foreign_resized/53.png',
    '389.png': '/foreign_resized/54.png',
    '390.png': '/foreign_resized/55.png',
    '391.png': '/foreign_resized/57.png',
    '392.png': '/foreign_resized/58.png',
    '393.png': '/foreign_resized/59.png',
    '394.png': '/foreign_resized/6.png',
    '395.png': '/foreign_resized/7.png',
    '396.png': '/foreign_resized/8.png',
    '397.png': '/foreign_resized/9.png'
}



# Step 3: Create a function to apply the mapping
def map_image_id(image_id):
    image_name = f"{image_id}.png"
    return mapping.get(image_name, '')

# Step 4: Apply mapping to comparisons data
comparisons['im1_path'] = comparisons['im1'].apply(map_image_id)
comparisons['im2_path'] = comparisons['im2'].apply(map_image_id)

# Debugging: Print a sample to verify the mapping
print("Mapping check:")
print(comparisons[['im1', 'im1_path', 'im2', 'im2_path']].head())

# Add full path to image names
comparisons['im1_full_path'] = comparison_images_path + '/' + comparisons['im1'].astype(str) + '.png'
comparisons['im2_full_path'] = comparison_images_path + '/' + comparisons['im2'].astype(str) + '.png'

# Step 5: Create label column and filter out pairs where w1 == w2
comparisons['label'] = comparisons.apply(lambda row: 1 if row['w1'] > row['w2'] else (-1 if row['w2'] > row['w1'] else None), axis=1)
comparisons = comparisons.dropna(subset=['label'])

# Convert the label column to integer
comparisons['label'] = comparisons['label'].astype(int)

# Adjust display options to show full path
pd.set_option('display.max_colwidth', None)

# Debugging: Print the comparisons with labels
print("Comparisons with labels:")
print(comparisons[['im1', 'im1_full_path', 'im2', 'im2_full_path', 'label']].head())

# Reset display options if needed
pd.reset_option('display.max_colwidth')

# Step 6: Filter comparisons to match train and test sets
train_images = set(train_means_list['image'])
test_images = set(test_list['image'])

# Debugging: Print the train and test image sets
print("Train images:", train_images)
print("Test images:", test_images)

# Check if both images in a pair exist in the same set (train or test)
train_pairs = comparisons[comparisons['im1_path'].isin(train_images) & comparisons['im2_path'].isin(train_images)]
test_pairs = comparisons[comparisons['im1_path'].isin(test_images) & comparisons['im2_path'].isin(test_images)]

# Debugging: Print the filtered train and test pairs
print("Filtered Train pairs:")
print(train_pairs[['im1', 'im1_full_path', 'im2', 'im2_full_path', 'label']].head())
print("Filtered Test pairs:")
print(test_pairs[['im1', 'im1_full_path', 'im2', 'im2_full_path', 'label']].head())

# Step 7: Save the results to CSV with full paths
train_pairs[['im1_full_path', 'im2_full_path', 'label']].to_csv('train_image_pairs.csv', index=False, header=['image_path1', 'image_path2', 'label'])
test_pairs[['im1_full_path', 'im2_full_path', 'label']].to_csv('test_image_pairs.csv', index=False, header=['image_path1', 'image_path2', 'label'])

# Verify exclusivity of image IDs between training and testing pairs
train_ids = set(train_pairs['im1_path'].tolist() + train_pairs['im2_path'].tolist())
test_ids = set(test_pairs['im1_path'].tolist() + test_pairs['im2_path'].tolist())

# Check for intersection
intersection = train_ids.intersection(test_ids)

# Print the result
if not intersection:
    print("No overlapping image IDs found between train and test pairs.")
else:
    print("Overlapping image IDs found between train and test pairs:", intersection)

# Additional Debugging: Print the overlapping IDs and corresponding rows
if intersection:
    print("Overlapping IDs details:")
    overlapping_train_pairs = train_pairs[train_pairs['im1_path'].isin(intersection) | train_pairs['im2_path'].isin(intersection)]
    overlapping_test_pairs = test_pairs[test_pairs['im1_path'].isin(intersection) | test_pairs['im2_path'].isin(intersection)]
    print("Train pairs with overlapping IDs:")
    print(overlapping_train_pairs)
    print("Test pairs with overlapping IDs:")
    print(overlapping_test_pairs)

# Additional debug: Check the original train and test lists for overlaps
train_image_set = set(train_means_list['image'])
test_image_set = set(test_list['image'])
original_intersection = train_image_set.intersection(test_image_set)

if not original_intersection:
    print("No overlapping image IDs found between train and test lists.")
else:
    print("Overlapping image IDs found in original train and test lists:", original_intersection)


#     # Step 3: Create a function to reverse the mapping
# def reverse_map_image_id(image_path):
#     for k, v in mapping.items():
#         if v == image_path:
#             return k
#     return ''

# # Step 4: Reverse map the image paths in train and test lists
# train_means_list['image'] = train_means_list['image'].apply(reverse_map_image_id)
# test_list['image'] = test_list['image'].apply(reverse_map_image_id)

# # Step 5: Apply mapping to comparisons data
# comparisons['im1_path'] = comparisons['im1'].apply(lambda x: comparison_images_path + reverse_map_image_id(mapping.get(f"{x}.png", '')))
# comparisons['im2_path'] = comparisons['im2'].apply(lambda x: comparison_images_path + reverse_map_image_id(mapping.get(f"{x}.png", '')))

# # Step 6: Create label column and filter out pairs where w1 == w2
# comparisons['label'] = comparisons.apply(lambda row: 1 if row['w1'] > row['w2'] else (-1 if row['w2'] > row['w1'] else None), axis=1)
# comparisons = comparisons.dropna(subset=['label'])

# # Step 7: Filter comparisons to match train and test sets
# train_images = set(train_means_list['image'])
# test_images = set(test_list['image'])

# train_pairs = comparisons[comparisons['im1'].isin(train_images) & comparisons['im2'].isin(train_images)]
# test_pairs = comparisons[comparisons['im1'].isin(test_images) & comparisons['im2'].isin(test_images)]

# # Step 8: Save the results to CSV with full paths
# train_pairs[['im1_path', 'im2_path', 'label']].to_csv(data_folder + 'train_image_pairs.csv', index=False, header=False)
# test_pairs[['im1_path', 'im2_path', 'label']].to_csv(data_folder + 'test_image_pairs.csv', index=False, header=False)

# # Verify exclusivity of image IDs between training and testing pairs
# train_ids = set(train_pairs['im1'].tolist() + train_pairs['im2'].tolist())
# test_ids = set(test_pairs['im1'].tolist() + test_pairs['im2'].tolist())

# # Check for intersection
# intersection = train_ids.intersection(test_ids)

# # Print the result
# if not intersection:
#     print("No overlapping image IDs found between train and test pairs.")
# else:
#     print("Overlapping image IDs found between train and test pairs:", intersection)
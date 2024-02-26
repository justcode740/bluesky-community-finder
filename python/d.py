import pandas as pd
pd.set_option('display.max_columns', None)  # Ensure all columns are shown
pd.set_option('display.width', 1000)  # Increase display width to avoid line wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column
# Define the file paths
posts_path = "bot-python/data/sampled_posts.csv"
follows_path = "bot-python/data/sampled_follows.csv"  # Adjusted assuming you have a follows CSV
likes_path = "bot-python/data/sampled_likes.csv"
profiles_path = "bot-python/data/sampled_profiles.csv"
reposts_path = "bot-python/data/sampled_reposts.csv"

# Load datasets
posts = pd.read_csv(posts_path)
follows = pd.read_csv(follows_path)  # Make sure to replace with the correct file name if different
likes = pd.read_csv(likes_path)  # Corrected the file name
profiles = pd.read_csv(profiles_path)
reposts = pd.read_csv(reposts_path)

# print(posts.head(10))

# print(profiles.head(10))
print(follows.head(10))
# print(posts['uri'].str.contains("at://did:plc:mjh257xyug2mpodoyy3he5ih/app.bsky.feed.post/3km5lx27r5u27").any())
# # Define a function to print details
# def print_details(csv_name, df):
#     print(f"\nColumns in {csv_name}: {df.columns.tolist()}")
#     print(f"Sample data from {csv_name}:\n{df.head(1)}\n")

# # Print details for each dataset
# print_details(posts_path, posts)
# print_details(follows_path, follows)
# print_details(likes_path, likes)
# print_details(profiles_path, profiles)
# print_details(reposts_path, reposts)


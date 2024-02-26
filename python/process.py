# # import pandas as pd
# pd.set_option('display.max_columns', None)  # Ensure all columns are shown
# pd.set_option('display.width', 1000)  # Increase display width to avoid line wrapping
# pd.set_option('display.max_colwidth', None)  # Show full content of each column
# # likes = pd.read_csv("bot-python/data/sampled_likes.csv")
# # posts = pd.read_csv("bot-python/data/sampled_posts.csv")
# # reposts = pd.read_csv("bot-python/data/sampled_reposts.csv")
# # profiles = pd.read_csv("bot-python/data/sampled_profiles.csv")

# # # Merge likes with posts to get liked posts' text and the creation date of the posts
# # likes_post = pd.merge(
# #     likes[['did', 'subject_uri']], 
# #     posts[['uri', 'text', 'created_at']], 
# #     left_on='subject_uri', 
# #     right_on='uri', 
# #     how='left'
# # ).drop(columns=['subject_uri', 'uri'])

# # # Rename columns for clarity
# # likes_post.rename(columns={
# #     'did': 'did', 
# #     'text': 'liked_posts_text', 
# #     'created_at': 'post_created_at'
# # }, inplace=True)

# # # Display the result
# # print(likes_post.head())

# # # Merge reposts with posts to get reposted posts' text and the creation date of the posts
# # repost_post = pd.merge(
# #     reposts[['did', 'subject_uri']], 
# #     posts[['uri', 'text', 'created_at']], 
# #     left_on='subject_uri', 
# #     right_on='uri', 
# #     how='left'
# # ).drop(columns=['subject_uri', 'uri'])

# # # Rename columns for clarity
# # repost_post.rename(columns={
# #     'did': 'did', 
# #     'text': 'reposted_posts_text', 
# #     'created_at': 'post_created_at'
# # }, inplace=True)

# # # Display the result
# # print(repost_post.head())

# # # Aggregate liked posts text by user did
# # liked_posts_agg = likes_post.groupby('did')['liked_posts_text'].apply(lambda x: ' '.join(x.dropna())).reset_index()

# # # Aggregate reposted posts text by user did
# # reposted_posts_agg = repost_post.groupby('did')['reposted_posts_text'].apply(lambda x: ' '.join(x.dropna())).reset_index()

# # # Aggregate original posts text by user did
# # original_posts_agg = posts.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index()



# # # did (of user), posts (in text/str), liked posts (in text/str), reposted posts (in text/str), profile (in text/str), and based on all these texts, compute tfidf and similarity, and find similar u# Start with the profiles DataFrame
# # final_df = profiles[['did', 'description']].rename(columns={'description': 'profile_text'})

# # # Perform sequential merges with likes, reposts, and original posts text
# # final_df = pd.merge(final_df, liked_posts_agg.rename(columns={'liked_posts_text': 'liked_text'}), on='did', how='left')
# # final_df = pd.merge(final_df, reposted_posts_agg.rename(columns={'reposted_posts_text': 'reposted_text'}), on='did', how='left')
# # final_df = pd.merge(final_df, original_posts_agg.rename(columns={'text': 'posts_text'}), on='did', how='left')

# # # Replace NaN with empty strings to avoid issues in further text processing
# # final_df.fillna('', inplace=True)

# # print(final_df.head())
# # # sers or posts that user might want.

# # # # user like post
# # # likes_post = pd.merge(likes, posts, left_on='subject_uri', right_on='uri', suffixes=('_like', '_post'))

# # # repost_post = pd.merge(reposts, posts, left_on='subject_uri', right_on='uri', suffix)

# # # print(likes_post.head(10))

# import datetime
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# pd.set_option('display.max_columns', None)  # Ensure all columns are shown
# pd.set_option('display.width', 1000)  # Increase display width to avoid line wrapping
# pd.set_option('display.max_colwidth', None)  # Show full content of each column
# # Load the data (Assuming CSV files are correctly formatted and paths are correct)
# likes = pd.read_csv("bot-python/data/sampled_likes.csv")
# posts = pd.read_csv("bot-python/data/sampled_posts.csv")
# reposts = pd.read_csv("bot-python/data/sampled_reposts.csv")
# profiles = pd.read_csv("bot-python/data/sampled_profiles.csv")

# # Merge operations
# likes_post = pd.merge(likes, posts[['uri', 'text']], left_on='subject_uri', right_on='uri', how='left')
# repost_post = pd.merge(reposts, posts[['uri', 'text']], left_on='subject_uri', right_on='uri', how='left')

# # Aggregation of texts
# liked_posts_agg = likes_post.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='liked_text')
# reposted_posts_agg = repost_post.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='reposted_text')
# original_posts_agg = posts.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='posts_text')

# # Ensure all dids are included
# all_dids = pd.concat([likes['did'], reposts['did'], posts['did'], profiles['did']]).unique()
# all_dids_df = pd.DataFrame(all_dids, columns=['did'])

# # Sequential merges
# final_df = all_dids_df.merge(profiles[['did', 'description']], on='did', how='left')
# final_df = final_df.merge(liked_posts_agg, on='did', how='left')
# final_df = final_df.merge(reposted_posts_agg, on='did', how='left')
# final_df = final_df.merge(original_posts_agg, on='did', how='left')


# # Fill NaNs with empty strings
# final_df.fillna('', inplace=True)

# print(final_df.columns)

# # Assuming weights: 1 for profiles, 1.5 for liked posts, 1.5 for reposted posts, and 2 for original posts
# # Here, we're just concatenating the texts with spaces in between. Adjust the weights as per your specific needs.
# # Concatenate texts with spaces, repeating the text to simulate weighting
# # Since we can't multiply by a float, we simulate "weighting" by repetition, but this approach has limitations
# final_df['weighted_text'] = (
#     final_df['description'].apply(lambda x: (x + ' ') * 1) +  # weight of 1 for description, repeated once
#     final_df['liked_text'].apply(lambda x: (x + ' ') * 2) +  # weight of 1.5 simulated by repeating twice
#     final_df['reposted_text'].apply(lambda x: (x + ' ') * 2) +  # weight of 1.5 simulated by repeating twice
#     final_df['posts_text'].apply(lambda x: (x + ' ') * 3)  # weight of 2 simulated by repeating thrice
# )


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel  # For faster cosine similarity computation on sparse matrices
# import joblib

# # Initialize the TF-IDF Vectorizer
# tfidf_vectorizer = TfidfVectorizer()

# # Combine all text fields into one for TF-IDF vectorization
# # Note: This simplifies the approach; you might want to keep them separate for more nuanced weight applications
# # final_df['combined_text'] = final_df.apply(lambda row: ' '.join([row['description'], row['liked_text'], row['reposted_text'], row['original_text']]), axis=1)

# # Compute TF-IDF matrix
# tfidf_matrix = tfidf_vectorizer.fit_transform(final_df['weighted_text'])
# timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# joblib.dump(tfidf_matrix, f'tfidf_matrix_{timestamp}.z')

# # Store TF-IDF vectorizer and matrix for later use (e.g., saving to disk, using joblib or pickle)
# def recommend_for_user(user_index, tfidf_matrix, top_k=5):
#     """
#     Generate recommendations for a user based on their combined text similarity to others.

#     Args:
#     - user_index: Index of the user in `final_df` for whom recommendations are being made.
#     - tfidf_matrix: Precomputed TF-IDF matrix of the combined text.
#     - top_k: Number of top recommendations to return.

#     Returns:
#     - DataFrame of top_k recommendations (user dids and similarity scores).
#     """
#     cosine_similarities = linear_kernel(tfidf_matrix[user_index:user_index+1], tfidf_matrix).flatten()
#     similar_indices = cosine_similarities.argsort()[:-top_k-1:-1]  # Get indices of top_k similarities, excluding the user themselves
#     similar_users = [(final_df.iloc[i]['did'], cosine_similarities[i]) for i in similar_indices if i != user_index]  # Exclude self
#     return pd.DataFrame(similar_users, columns=['did', 'similarity'])


# def search_and_recommend(keyword, tfidf_vectorizer, tfidf_matrix, final_df, top_k=5):
#     """
#     Search for and recommend posts and users related to a specific keyword.

#     Args:
#     - keyword (str): The keyword to search for.
#     - tfidf_vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
#     - tfidf_matrix (sparse matrix): The TF-IDF matrix for all combined texts.
#     - final_df (DataFrame): DataFrame containing user DIDs and their texts.
#     - top_k (int): Number of top results to return.

#     Returns:
#     - DataFrame: Top posts related to the keyword.
#     - DataFrame: Top users related to the keyword.
#     """
#     # Transform the keyword into its TF-IDF vector
#     keyword_vector = tfidf_vectorizer.transform([keyword])
    
#     # Compute cosine similarity between the keyword vector and all post/user vectors
#     cosine_similarities = linear_kernel(keyword_vector, tfidf_matrix).flatten()
    
#     # Get top_k results
#     similar_indices = cosine_similarities.argsort()[-top_k:][::-1]
    
#     # Extract DID and similarity scores for top users
#     similar_users = [(final_df.iloc[i]['did'], cosine_similarities[i]) for i in similar_indices]
#     similar_users_df = pd.DataFrame(similar_users, columns=['DID', 'Similarity'])

#     # Extract top posts related to the keyword
#     top_posts_indices = similar_indices[:top_k]  # Adjust as needed to get top posts
#     top_posts = [final_df.iloc[i]['posts_text'] for i in top_posts_indices]
#     top_posts_df = pd.DataFrame(top_posts, columns=['Post'])

#     return similar_users_df, top_posts_df

# # Example usage

# if __name__ == "__main__":
#     keyword = "rust"  # Example keyword
#     similar_users_df, top_posts_df = search_and_recommend(keyword, tfidf_vectorizer, tfidf_matrix, final_df, top_k=5)

#     print("Users related to the keyword:")
#     print(similar_users_df)
#     print("\nPosts related to the keyword:")
#     print(top_posts_df)


# # Example usage for a user at index 0
# # recommendations = recommend_for_user(0, tfidf_matrix, top_k=5)
# # print(recommendations)



# # Example of applying weights during text concatenation
# # Assuming weights: profile - 1, posts - 2, liked posts - 1.5, reposted posts - 1.5
# # final_df['weighted_texts'] = (final_df['profile_text'] + ' ' +
# #                               final_df['posts_text'] * 2 + ' ' +  # Assuming string multiplication for illustration
# #                               final_df['liked_text'] * 1.5 + ' ' +
# #                               final_df['reposted_text'] * 1.5)

# # Proceed with TF-IDF vectorization and similarity computation on 'weighted_texts'

# # # Example TF-IDF computation and similarity (for one text type, repeat for others and combine as needed)
# # tfidf_vectorizer = TfidfVectorizer()
# # tfidf_matrix = tfidf_vectorizer.fit_transform(final_df['description'])  # Example with profile descriptions

# # # Compute similarity (example within the same text type, extend as needed)
# # cosine_sim = cosine_similarity(tfidf_matrix)

# # # Store similarity in a DataFrame
# # similarity_df = pd.DataFrame(cosine_sim, index=final_df['did'], columns=final_df['did'])

# # # Example saving the DataFrame
# # similarity_df.to_csv('user_similarity.csv')

# # print("TF-IDF and similarity computation completed.")

import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
pd.set_option('display.max_columns', None)  # Ensure all columns are shown
pd.set_option('display.width', 1000)  # Increase display width to avoid line wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column
# Load the data (Assuming CSV files are correctly formatted and paths are correct)
likes = pd.read_csv("bot-python/data/sampled_likes.csv")
posts = pd.read_csv("bot-python/data/sampled_posts.csv")
reposts = pd.read_csv("bot-python/data/sampled_reposts.csv")
profiles = pd.read_csv("bot-python/data/sampled_profiles.csv")

def preprocess_data(profile_weight=1, liked_posts_weight=1.5, reposted_posts_weight=1.5, original_posts_weight=2):
    # Load the data (Assuming CSV files are correctly formatted and paths are correct)
    likes = pd.read_csv("bot-python/data/sampled_likes.csv")
    posts = pd.read_csv("bot-python/data/sampled_posts.csv")
    reposts = pd.read_csv("bot-python/data/sampled_reposts.csv")
    profiles = pd.read_csv("bot-python/data/sampled_profiles.csv")
    posts['cid'] = posts['uri'].apply(lambda x: x.split('/')[-1])


    # Merge operations
    likes_post = pd.merge(likes, posts[['uri', 'text']], left_on='subject_uri', right_on='uri', how='left')
    repost_post = pd.merge(reposts, posts[['uri', 'text']], left_on='subject_uri', right_on='uri', how='left')

    # Aggregation of texts
    liked_posts_agg = likes_post.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='liked_text')
    reposted_posts_agg = repost_post.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='reposted_text')
    original_posts_agg = posts.groupby('did').apply(lambda x: pd.Series({
        'posts_text': ' '.join(x['text'].dropna()),
        'cids': clean_and_unique_cid(' '.join(x['cid'].dropna()))
    })).reset_index()

    # Ensure all dids are included
    all_dids = pd.concat([likes['did'], reposts['did'], posts['did'], profiles['did']]).unique()
    all_dids_df = pd.DataFrame(all_dids, columns=['did'])

    # Sequential merges
    final_df = all_dids_df.merge(profiles[['did', 'description']], on='did', how='left')
    final_df = final_df.merge(liked_posts_agg, on='did', how='left')
    final_df = final_df.merge(reposted_posts_agg, on='did', how='left')
    final_df = final_df.merge(original_posts_agg, on='did', how='left')

    # Fill NaNs with empty strings
    final_df.fillna('', inplace=True)

    # Concatenate texts with adjustable weights
    final_df['weighted_text'] = (
        final_df['description'].apply(lambda x: (x + ' ') * int(profile_weight)) +
        final_df['liked_text'].apply(lambda x: (x + ' ') * int(liked_posts_weight)) +
        final_df['reposted_text'].apply(lambda x: (x + ' ') * int(reposted_posts_weight)) +
        final_df['posts_text'].apply(lambda x: (x + ' ') * int(original_posts_weight))
    )


    return final_df

def preprocess_data_large(profile_weight=1, liked_posts_weight=1.5, reposted_posts_weight=1.5, original_posts_weight=2):
    # Load the data (Assuming CSV files are correctly formatted and paths are correct)
    likes = pd.read_csv("bot-python/data/likes.csv")
    posts = pd.read_csv("bot-python/data/posts.csv")
    reposts = pd.read_csv("bot-python/data/reposts.csv")
    profiles = pd.read_csv("bot-python/data/profiles.csv")
    posts['cid'] = posts['uri'].apply(lambda x: x.split('/')[-1])

    # Example usage of the adjusted function
    profiles, posts, likes, reposts = preprocess_data_select_top_users(profiles, posts, likes, reposts)

    


    # Merge operations
    likes_post = pd.merge(likes, posts[['uri', 'text']], left_on='subject_uri', right_on='uri', how='left')
    repost_post = pd.merge(reposts, posts[['uri', 'text']], left_on='subject_uri', right_on='uri', how='left')

    # Aggregation of texts
    liked_posts_agg = likes_post.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='liked_text')
    reposted_posts_agg = repost_post.groupby('did')['text'].apply(lambda x: ' '.join(x.dropna())).reset_index(name='reposted_text')
    original_posts_agg = posts.groupby('did').apply(lambda x: pd.Series({
    'posts_text': ' '.join(x['text'].dropna()),
    'cids': ' '.join(x['cid'].dropna())
    })).reset_index()

    # Ensure all dids are included
    all_dids = pd.concat([likes['did'], reposts['did'], posts['did'], profiles['did']]).unique()
    all_dids_df = pd.DataFrame(all_dids, columns=['did'])

    # Sequential merges
    final_df = all_dids_df.merge(profiles[['did', 'description']], on='did', how='left')
    final_df = final_df.merge(liked_posts_agg, on='did', how='left')
    final_df = final_df.merge(reposted_posts_agg, on='did', how='left')
    final_df = final_df.merge(original_posts_agg, on='did', how='left')

    # Fill NaNs with empty strings
    final_df.fillna('', inplace=True)

    # Concatenate texts with adjustable weights
    final_df['weighted_text'] = (
        final_df['description'].apply(lambda x: (x + ' ') * int(profile_weight)) +
        final_df['liked_text'].apply(lambda x: (x + ' ') * int(liked_posts_weight)) +
        final_df['reposted_text'].apply(lambda x: (x + ' ') * int(reposted_posts_weight)) +
        final_df['posts_text'].apply(lambda x: (x + ' ') * int(original_posts_weight))
    )


    return final_df

def compute_tfidf(final_df):
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Compute TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(final_df['weighted_text'])
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    joblib.dump(tfidf_matrix, f'tfidf_matrix_large_{timestamp}.z')
    joblib.dump(tfidf_vectorizer, f'tfidf_vectorizer_large_{timestamp}.joblib')

    return tfidf_vectorizer, tfidf_matrix

def preprocess_data_select_top_users(profiles, posts, likes, reposts, top_k=50000):
     # Aggregate activity counts (posts, likes, reposts) for each user
    activity_counts = (
        posts['did'].value_counts().rename('posts_count')
        .add(likes['did'].value_counts().rename('likes_count'), fill_value=0)
        .add(reposts['did'].value_counts().rename('reposts_count'), fill_value=0)
    )

    # Select top_k users based on activity
    top_users = activity_counts.nlargest(top_k).index

    # Filter data for top users
    profiles = profiles[profiles['did'].isin(top_users)]
    posts = posts[posts['did'].isin(top_users)]
    likes = likes[likes['did'].isin(top_users)]
    reposts = reposts[reposts['did'].isin(top_users)]

    return profiles, posts, likes, reposts

# Further processing as needed, e.g., merging, TF-IDF computation
def clean_and_unique_cid(cid_str):
    # Split the CID string by spaces to get individual CIDs, remove duplicates by converting to a set, then join back
    return ' '.join(set(cid_str.split()))

def search_and_recommend(keyword, tfidf_vectorizer, tfidf_matrix, final_df, top_k=20):
    """
    Search for and recommend posts and users related to a specific keyword.

    Args:
    - keyword (str): The keyword to search for.
    - tfidf_vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    - tfidf_matrix (sparse matrix): The TF-IDF matrix for all combined texts.
    - final_df (DataFrame): DataFrame containing user DIDs and their texts.
    - top_k (int): Number of top results to return.

    Returns:
    - DataFrame: Top posts related to the keyword.
    - DataFrame: Top users related to the keyword.
    """
    # Transform the keyword into its TF-IDF vector
    keyword_vector = tfidf_vectorizer.transform([keyword])
    
    # Compute cosine similarity between the keyword vector and all post/user vectors
    cosine_similarities = linear_kernel(keyword_vector, tfidf_matrix).flatten()
    
    # Get top_k results
    similar_indices = cosine_similarities.argsort()[-top_k:][::-1]
    
    # Extract DID and similarity scores for top users
    similar_users = [(final_df.iloc[i]['did'], cosine_similarities[i]) for i in similar_indices]
    similar_users_df = pd.DataFrame(similar_users, columns=['DID', 'Similarity'])

    top_posts_indices = similar_indices[:top_k]  # Adjust as needed to get top posts
    top_posts = [{
        'Post': final_df.iloc[i]['posts_text'],
        'CID': final_df.iloc[i]['cids'],  # Assuming 'cids' contains the concatenated cids
        'DID': final_df.iloc[i]['did']
    } for i in top_posts_indices]
    top_posts_df = pd.DataFrame(top_posts)

     # Filter out results with 'NaN' or None in 'Post' or 'CID'
    # filtered_results = [result for result in top_posts_df if result['Post'] is not None and result['CID'] is not None]
    
    # Ensure unique CIDs and clean up CID strings
    # for result in filtered_results:
    #     result['CID'] = clean_and_unique_cid(result['CID'])

    top_posts_df = top_posts_df.dropna(subset=['Post', 'CID'])
    
    # Ensure unique CIDs by cleaning up CID strings
    top_posts_df['CID'] = top_posts_df['CID'].apply(clean_and_unique_cid)

    return similar_users_df, top_posts_df

if __name__ == "__main__":
    import os
    final_df_large_path = "final_df_large_2.csv"

    # Check if final_df_large exists
    if os.path.exists(final_df_large_path):
        # print(f"Loading existing DataFrame from {final_df_large_path}")
        final_df = pd.read_csv(final_df_large_path)
    else:
        print("Preprocessing data to create final_df_large...")
        # Assuming preprocess_data_large() is defined and loads data correctly
        # final_df = preprocess_data_large()

        final_df = preprocess_data_large()
        final_df.to_csv(final_df_large_path, index=False)
        print(f"DataFrame saved to {final_df_large_path}")
    # final_df.to_csv("final_df_large.csv", index=False)
    
    # final_df.head(10)
    print(final_df.columns)
    print(final_df['cids'].head(10))
    tfidf_vectorizer, tfidf_matrix = compute_tfidf(final_df)
    keyword = "rust"  # Example keyword
    similar_users_df, top_posts_df = search_and_recommend(keyword, tfidf_vectorizer, tfidf_matrix, final_df, top_k=5)

    print("Users related to the keyword:")
    print(similar_users_df)
    print("\nPosts related to the keyword:")
    print(top_posts_df)

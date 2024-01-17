
import pandas as pd
import pandas as pd
import re
from collections import Counter


# Function to create a smaller db for testing
# def create_smaller_db(doc, rows):
#     df = pd.read_csv(doc)
#     df_smaller = df.head(rows)
#     df_smaller.to_csv(doc + str(rows), index=False)


def create_smaller_db(df, rows, positive_percentage, negative_percentage):
    # Shuffle the DataFrame rows
    df_smaller = df.head(rows)
    df_shuffled = df_smaller.sample(frac=1, random_state=28)
    
    # Calculate the desired subset sizes based on percentages
    positive_count = int(len(df) * positive_percentage)
    negative_count = int(len(df) * negative_percentage)
    
    # Filter positive and negative labels
    positive_subset = df_shuffled[df_shuffled['label'] == 1].head(positive_count)
    negative_subset = df_shuffled[df_shuffled['label'] == 0].head(negative_count)
    
    # Concatenate the subsets
    subset = pd.concat([positive_subset, negative_subset])
    
    return subset


#to normalize clinical note text (ex. remove symbols, abbreviations, etc.)
#modified version to make changes in each row
def normalize_df_column(df, col_name='text', filtered_col_name='filtered_text'):
    df[filtered_col_name] = df[col_name].copy()  # Create a new column for filtered text

    for i in range(len(df)):
        text = df.loc[i, col_name]

        text = re.sub(r"_+", "unk", text)        # replace all consecutive underscores with unknown
        text = re.sub(r"\"", "'", text)           # replace " (double quote) with ' (single quote)
        text = re.sub(r"\s{2,}", " ", text)        # replaces consecutive blank space 
        text = re.sub(r" +", " ", text)
        text = re.sub(r"#", "", text)             # replaces hashtags with space
        text = re.sub(r"@", "at", text)            # replaces @ with 'at'
        text = re.sub(r" pt ", " patient ", text) 
        text = re.sub(r" Pt ", " patient ", text) 
        df.loc[i, filtered_col_name] = text

    return df



#to extract most recurrent categories from discharge notes based on certain criteria 
def extract_recurrent_note_categories(df):
    # pattern for 1 to 5 words followed by ':' or nothing, and a newline
    pattern = r'\b(\w+(\s\w+){0,4})(?:\:)?\n'
    sentences = []
    for text in df['text']:
        # find all matches of the pattern
        matches = re.findall(pattern, text, re.MULTILINE)
        # extract the sentence from each match (first group of the regex)
        matches = [match[0] for match in matches]
        
        # filter out not wanted sentences containing numbers, articles, pronouns, preposition etc.
        not_wanted = r'\b(not|none|to|on|or|a|an|any|and|the|she|he|his|her|your|it|I|you|we|they|there|may|this|that|with|by|rrr|of|ml|mmm|______|[0-9]+)\b'
        matches = [match for match in matches if not re.search(not_wanted, match, re.IGNORECASE)]
        
        # filter out sentences that contain only one character as a word
        matches = [match for match in matches if not any(len(word) <= 1 for word in match.split())]
        
        # add filtered matches to list only if first letter of first word is capitalized
        sentences += [match for match in matches if match.split()[0][0].isupper() or match.split()[0].isupper()]
        
    # Get only recurrent sentences (occuring more than once)
    counter = Counter(sentences)
    recurrent_sentences = [sentence for sentence, count in counter.items() if count > 10000]
    return recurrent_sentences


#to filter text by splitting it between accepted text after accepted list and reject list
def filter_text(df, accept_list, reject_list):
    pattern_accept = "|".join(accept_list)
    pattern_reject = "|".join(reject_list)
    df["filtered_text"] = df["text"].apply(lambda x: re.split(pattern_reject, re.split(pattern_accept, x, maxsplit=1)[-1])[0] if any(substring in x for substring in accept_list) else '')

    return df

#to add labels 1/0 based on icd_code
def from_code_add_label_col(df, code_list):
    df = df.copy()
    df['label'] = df['icd_code'].apply(lambda i: 1 if i in code_list else 0)
    return df

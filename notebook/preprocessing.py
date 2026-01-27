import re 

# lowering the text
def to_lowercase(text):
    
    return text.lower()  

# removing the html tags
def remove_html_tags(text):

    pattern = re.compile('<.*?>')

    return pattern.sub('', text)

# remove punctuation
def remove_punctuation(text):

    return re.sub(r'[^a-z0-9\s]', ' ', text).strip()
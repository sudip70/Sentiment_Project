emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', ':-(': 'sad', ':-<': 'sad', 
    ':P': 'raspberry', ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed', 
    ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy', '@@': 'eyeroll', 
    ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused', '<(-_-)>': 'robot', 'd[-_-]b': 'dj', 
    ":'-)": 'sadsmile', ';)': 'wink', ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}

abbreviations = {
    "$": " dollar ", "€": " euro ", "4ao": "for adults only", "a.m": "before midday", "afaik": "as far as I know", 
    "app": "application", "asap": "as soon as possible", "atm": "at the moment", "brb": "be right back", 
    "btw": "by the way", "cu": "see you", "faq": "frequently asked questions", "fyi": "for your information", 
    "idk": "I do not know", "imho": "in my humble opinion", "lol": "laughing out loud", "omg": "oh my god", 
    "omw": "on my way", "ppl": "people", "smh": "shake my head", "tbh": "to be honest", "ttyl": "talk to you later", 
    "u": "you", "w/": "with", "w/o": "without", "wtf": "what the f***", "wyd": "what you doing"
}

# Data cleaning class to clean the input data
class DataCleaning:
    def __init__(self):
        self.stop_word = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()  # For stemming
        self.lemmatizer = WordNetLemmatizer()  # For lemmatization
    
    def expand_contractions(self, text):
        return contractions.fix(text)
    
    def replace_emojis(self, text):
        for emoji, meaning in emojis.items():
            text = text.replace(emoji, meaning)
        return text
    
    def replace_abbreviations(self, text):
        for abbr, full_form in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
        return text
    
    def clean_text(self, text):
        text = text.lower()
        text = self.expand_contractions(text)
        text = self.replace_emojis(text)
        text = self.replace_abbreviations(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_word]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

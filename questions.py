import nltk
nltk.download('punkt')
nltk.download('stopwords')
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)
    while True:

        # Prompt user for query
        query = set(tokenize(input("Query (or hit X to exit): ")))

        # Determine top file matches according to TF-IDF
        filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

        if len(query) == 1 and (list(query)[0] == 'exit' or list(query)[0] == 'x'):
            print('Ok, see you later!!!')
            sys.exit()

        elif len(query) < 1:
            print(f'Please enter a valid question!\n')

        else:

        # Extract sentences from top files
            sentences = dict()
            for filename in filenames:
                for passage in files[filename].split("\n"):
                    for sentence in nltk.sent_tokenize(passage):
                        tokens = tokenize(sentence)
                        if tokens:
                            sentences[sentence] = tokens

            # Compute IDF values across sentences
            idfs = compute_idfs(sentences)

            # Determine top sentence matches
            matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
            for match in matches:
                print(f'{match}\n')


def load_files(directory):
   

    topics  = {}
    path_to_dir  = os.path.join(".",f"{directory}")
    
    for file in os.listdir(path_to_dir):
        path_to_file = os.path.join(path_to_dir,file)
        with open(path_to_file,"r",encoding="utf8") as f:
            string = f.read()

        topics[file[:-4]] = string
    return topics
    


def tokenize(document):
    
    tokens = [word.lower() for word in nltk.word_tokenize(document)]
    filtered = []

    stopwords = nltk.corpus.stopwords.words("english")
    punct = [punct for punct in string.punctuation]

    for word in tokens:
        if word in stopwords:
            continue
        elif word in punct:
            continue
        else:
            filtered.append(word)




    return(filtered)

def compute_idfs(documents):
    
    numDict = len(documents)
    presence = {}
    idfs = {}

    for doc in documents:
        for word in set(documents[doc]):
            if word in presence.keys():
                presence[word] += 1
            else:
                presence[word]  = 1

    for word in presence:
        idf = 1 + (math.log(numDict/(presence[word])))
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    
    tfidfs = {}
    for file in files:
        tfidfs[file] = 0
        tokens_in_file = len(files[file])
        for word in query:
            if word in files[file]:
                frequency = files[file].count(word) + 1
            else:
                frequency = 1
            tf = frequency/tokens_in_file
            if word in idfs.keys():
                idf = idfs[word]
            else:
                idf = 1
            tfidfs[file] += idf * tf

    sorted_list = sorted(tfidfs, key=tfidfs.get, reverse= True)
    topFiles = sorted_list[:n]

    return topFiles


def top_sentences(query, sentences, idfs, n):
    
    sentence_stats = {}
    for sentence in sentences:
        sentence_stats[sentence] = {}
        sentence_stats[sentence]['idf'] = 0
        sentence_stats[sentence]['word_count'] = 0

        senlength = len(sentences[sentence])
        for word in query:
            if word in sentences[sentence]:
                sentence_stats[sentence]['idf'] += idfs[word]
                sentence_stats[sentence]['word_count'] += 1
        sentence_stats[sentence]['QTD'] = float(sentence_stats[sentence]['word_count'] / senlength)
        
    sorted_list = sorted(sentence_stats.keys(), key = lambda sentence: (sentence_stats[sentence]['idf'], sentence_stats[sentence]['QTD']), reverse=True)

    topSens = sorted_list[:n]
    return topSens



if __name__ == "__main__":
    main()
